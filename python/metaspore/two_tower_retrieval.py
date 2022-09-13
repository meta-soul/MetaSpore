#
# Copyright 2022 DMetaSoul
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import json
import torch
import numpy
import faiss
import pyspark
from . import _metaspore
from .estimator import PyTorchAgent
from .estimator import PyTorchLauncher
from .estimator import PyTorchModel
from .estimator import PyTorchEstimator
from .metric import ModelMetric

class TwoTowerRetrievalModule(torch.nn.Module):
    def __init__(self, user_module, item_module, similarity_module):
        super().__init__()
        if not isinstance(user_module, torch.nn.Module):
            raise TypeError(f"user_module must be torch.nn.Module; {user_module!r} is invalid")
        if not isinstance(item_module, torch.nn.Module):
            raise TypeError(f"item_module must be torch.nn.Module; {item_module!r} is invalid")
        if not isinstance(similarity_module, torch.nn.Module):
            raise TypeError(f"similarity_module must be torch.nn.Module; {similarity_module!r} is invalid")
        self._user_module = user_module
        self._item_module = item_module
        self._similarity_module = similarity_module

    @property
    def user_module(self):
        return self._user_module

    @property
    def item_module(self):
        return self._item_module

    @property
    def similarity_module(self):
        return self._similarity_module

    def forward(self, x):
        user_emb = self._user_module(x)
        item_emb = self._item_module(x)
        sim = self._similarity_module(user_emb, item_emb)
        return sim

class TwoTowerIndexBuilder(object):
    def __init__(self, agent):
        self._agent = agent

    @property
    def agent(self):
        return self._agent

    @property
    def index_type(self):
        if not hasattr(self, '_index_type'):
            from .name_utils import simplify_name
            from .name_utils import to_lower_snake_case
            self_class_name = self.__class__.__name__
            base_class_name = TwoTowerIndexBuilder.__name__
            name = simplify_name(self_class_name, base_class_name)
            self._index_type = to_lower_snake_case(name)
        return self._index_type

    @property
    def index_meta_dir(self):
        if not hasattr(self, '_index_meta_dir'):
            from .url_utils import use_s3
            model_in_path = self.agent.model_in_path
            self._index_meta_dir = use_s3('%s%s/' % (model_in_path, self.index_type))
        return self._index_meta_dir

    @property
    def index_meta_file_name(self):
        return 'index_meta.json'

    @property
    def index_meta_path(self):
        if not hasattr(self, '_index_meta_path'):
            dir_path = self.index_meta_dir
            file_name = self.index_meta_file_name
            self._index_meta_path = '%s%s' % (dir_path, file_name)
        return self._index_meta_path

    @property
    def item_ids_dir(self):
        if not hasattr(self, '_item_ids_dir'):
            from .url_utils import use_s3
            model_in_path = self.agent.model_in_path
            self._item_ids_dir = use_s3('%s%s/item_ids/' % (model_in_path, self.index_type))
        return self._item_ids_dir

    @property
    def item_ids_partition_file_name(self):
        if not hasattr(self, '_item_ids_partition_file_name'):
            rank = self.agent.rank
            worker_count = self.agent.worker_count
            self._item_ids_partition_file_name = 'part_%d_%d.dat' % (worker_count, rank)
        return self._item_ids_partition_file_name

    @property
    def item_ids_partition_path(self):
        if not hasattr(self, '_item_ids_partition_path'):
            dir_path = self.item_ids_dir
            file_name = self.item_ids_partition_file_name
            self._item_ids_partition_path = '%s%s' % (dir_path, file_name)
        return self._item_ids_partition_path

    @property
    def have_item_ids_partition_files(self):
        return self.agent.enable_item_id_mapping or self.agent.output_item_embeddings

    def _make_index_meta(self):
        meta_version = 1
        index_type = self.index_type
        partition_count = self.agent.worker_count
        item_embedding_size = self.agent.item_embedding_size
        item_ids_field_delimiter = self.agent.item_ids_field_delimiter
        item_ids_value_delimiter = self.agent.item_ids_value_delimiter
        enable_item_id_mapping = self.agent.enable_item_id_mapping
        output_item_embeddings = self.agent.output_item_embeddings
        retrieval_item_count = self.agent.retrieval_item_count
        have_item_ids_partition_files = self.have_item_ids_partition_files
        item_id_column_type = self.item_id_column_type
        meta = {
            'meta_version' : meta_version,
            'index_type' : index_type,
            'partition_count' : partition_count,
            'item_embedding_size' : item_embedding_size,
            'item_ids_field_delimiter' : item_ids_field_delimiter,
            'item_ids_value_delimiter' : item_ids_value_delimiter,
            'enable_item_id_mapping': enable_item_id_mapping,
            'output_item_embeddings' : output_item_embeddings,
            'retrieval_item_count' : retrieval_item_count,
            'have_item_ids_partition_files' : have_item_ids_partition_files,
            'item_id_column_type' : item_id_column_type,
        }
        return meta

    def _output_index_meta(self):
        meta = self._make_index_meta()
        string = json.dumps(meta, separators=(',', ': '), indent=4)
        data = (string + '\n').encode('utf-8')
        _metaspore.ensure_local_directory(self.index_meta_dir)
        _metaspore.stream_write_all(self.index_meta_path, data)

    def _load_index_meta(self):
        data = _metaspore.stream_read_all(self.index_meta_path)
        string = data.decode('utf-8')
        meta = json.loads(string)
        return meta

    def get_index_meta(self):
        if not hasattr(self, '_index_meta'):
            self._index_meta = self._load_index_meta()
        return self._index_meta

    def _open_item_ids_partition_output_stream(self):
        print("Open %s item ids mapping partition file: %s" % (self.index_type, self.item_ids_partition_path))
        _metaspore.ensure_local_directory(self.item_ids_dir)
        self._item_ids_partition_output_stream = _metaspore.OutputStream(self.item_ids_partition_path)

    def _close_item_ids_partition_output_stream(self):
        self._item_ids_partition_output_stream = None

    def _convert_spark_item_id_column_type(self, data_type):
        from pyspark.sql.types import StringType
        from pyspark.sql.types import IntegerType
        from pyspark.sql.types import LongType
        if isinstance(data_type, StringType):
            return 'string'
        elif isinstance(data_type, (IntegerType, LongType)):
            return 'int64'
        else:
            message = f"item id column type must be one of: int, long, string; "
            message += f"{data_type!r} is not supported"
            raise RuntimeError(message)

    @property
    def item_id_column_type(self):
        if not hasattr(self, '_item_id_column_type'):
            item_id_column_name = self.agent.item_id_column_name
            item_ids_column_indices = self.agent.item_ids_column_indices
            item_ids_column_names = self.agent.item_ids_column_names
            enable_item_id_mapping = self.agent.enable_item_id_mapping
            if enable_item_id_mapping:
                self._item_id_column_type = 'int64'
            elif item_ids_column_indices is not None:
                assert len(item_ids_column_indices) == 1
                column_index = item_ids_column_indices[0]
                item_dataset = self.agent.dataset
                if column_index >= len(item_dataset.schema):
                    message = f"column_index {column_index} in item_ids_column_indices is out of range; "
                    message += f"item_dataset has only {len(item_dataset.schema)} columns"
                    raise RuntimeError(message)
                field = item_dataset.schema[column_index]
                self._item_id_column_type = self._convert_spark_item_id_column_type(field.dataType)
            else:
                assert item_ids_column_names is not None
                assert len(item_ids_column_names) == 1
                column_name = item_ids_column_names[0]
                item_dataset = self.agent.dataset
                if column_name not in item_dataset.columns:
                    message = f"column_name {column_name!r} in item_ids_column_names is not found "
                    message += f"in the columns of item_dataset: {item_dataset.columns!r}"
                    raise RuntimeError(message)
                field = item_dataset.schema[column_name]
                self._item_id_column_type = self._convert_spark_item_id_column_type(field.dataType)
        return self._item_id_column_type

    def get_item_id_ndarray(self, minibatch):
        item_id_column_name = self.agent.item_id_column_name
        item_ids_column_indices = self.agent.item_ids_column_indices
        item_ids_column_names = self.agent.item_ids_column_names
        enable_item_id_mapping = self.agent.enable_item_id_mapping
        if enable_item_id_mapping:
            column = minibatch.loc[:, item_id_column_name]
        elif item_ids_column_indices is not None:
            assert len(item_ids_column_indices) == 1
            column_index = item_ids_column_indices[0]
            column = minibatch.iloc[:, column_index]
        else:
            assert item_ids_column_names is not None
            assert len(item_ids_column_names) == 1
            column_name = item_ids_column_names[0]
            column = minibatch.loc[:, column_name]
        id_ndarray = column.values
        return id_ndarray

    def make_item_ids_mapping_batch(self, minibatch, embeddings, id_ndarray):
        item_ids_column_indices = self.agent.item_ids_column_indices
        item_ids_column_names = self.agent.item_ids_column_names
        item_ids_field_delimiter = self.agent.item_ids_field_delimiter
        item_ids_value_delimiter = self.agent.item_ids_value_delimiter
        enable_item_id_mapping = self.agent.enable_item_id_mapping
        output_item_embeddings = self.agent.output_item_embeddings
        ids_data = ''
        for i in range(len(id_ndarray)):
            ids_data += str(id_ndarray[i])
            ids_data += item_ids_field_delimiter
            if enable_item_id_mapping:
                if item_ids_column_indices is not None:
                    for j, index in enumerate(item_ids_column_indices):
                        if j > 0:
                            ids_data += item_ids_value_delimiter
                        field = minibatch.iloc[i, index]
                        if field is not None:
                            ids_data += str(field)
                else:
                    for j, column_name in enumerate(item_ids_column_names):
                        if j > 0:
                            ids_data += item_ids_value_delimiter
                        field = minibatch.loc[i, column_name]
                        if field is not None:
                            ids_data += str(field)
            if output_item_embeddings:
                ids_data += item_ids_field_delimiter
                for k, value in enumerate(embeddings[i]):
                    if k > 0:
                        ids_data += item_ids_value_delimiter
                    ids_data += str(value)
            ids_data += '\n'
        return ids_data

    def output_item_ids_mapping_batch(self, ids_data):
        ids_data = ids_data.encode('utf-8')
        self._item_ids_partition_output_stream.write(ids_data)

    def _make_item_ids_schema(self, meta):
        from pyspark.sql.types import StructType
        from pyspark.sql.types import StructField
        from pyspark.sql.types import LongType
        from pyspark.sql.types import StringType
        from pyspark.sql.types import FloatType
        from pyspark.sql.types import ArrayType
        enable_item_id_mapping = meta['enable_item_id_mapping']
        output_item_embeddings = meta['output_item_embeddings']
        item_id_column_type = meta['item_id_column_type']
        fields = []
        if enable_item_id_mapping:
            fields.append(StructField('id', LongType(), True))
            fields.append(StructField('name', StringType(), True))
        else:
            id_column_type = LongType() if item_id_column_type == 'int64' else StringType()
            fields.append(StructField('id', id_column_type, True))
        if output_item_embeddings:
            fields.append(StructField('item_embedding', ArrayType(FloatType()), True))
        schema = StructType(fields)
        return schema

    def load_item_ids(self):
        from .input import read_s3_csv
        meta = self.get_index_meta()
        item_ids_field_delimiter = meta['item_ids_field_delimiter']
        item_ids_value_delimiter = meta['item_ids_value_delimiter']
        have_item_ids_partition_files = meta['have_item_ids_partition_files']
        if not have_item_ids_partition_files:
            return None
        spark = self.agent.spark_session
        schema = self._make_item_ids_schema(meta)
        df = read_s3_csv(spark, self.item_ids_dir,
                         schema=schema,
                         delimiter=item_ids_field_delimiter,
                         multivalue_delimiter=item_ids_value_delimiter)
        return df

    def search_item_embedding_batch(self, embeddings):
        raise NotImplementedError

    def _copy_index_files(self):
        if self.agent.model_export_path is not None:
            from .url_utils import use_s3
            from .file_utils import copy_dir
            src_path = use_s3('%s%s/' % (self.agent.model_in_path, self.index_type))
            dst_path = use_s3('%s%s/' % (self.agent.model_export_path, self.index_type))
            copy_dir(src_path, dst_path)

    def begin_creating_index(self):
        pass

    def end_creating_index(self):
        self._output_index_meta()
        self._copy_index_files()

    def begin_creating_index_partition(self):
        if self.have_item_ids_partition_files:
            self._open_item_ids_partition_output_stream()

    def end_creating_index_partition(self):
        if self.have_item_ids_partition_files:
            self._close_item_ids_partition_output_stream()

    def distribute_state(self):
        pass

    def begin_querying_index(self):
        pass

    def end_querying_index(self):
        pass

class TwoTowerFaissIndexBuilder(TwoTowerIndexBuilder):
    def __init__(self, agent):
        super().__init__(agent)
        self.faiss_index_description = getattr(agent, 'faiss_index_description', 'Flat')
        self.faiss_metric_type = getattr(agent, 'faiss_metric_type', 'METRIC_INNER_PRODUCT')
        if not isinstance(self.faiss_index_description, str) or not self.faiss_index_description:
            raise TypeError(f"faiss_index_description must be non-empty string; {self.faiss_index_description!r} is invalid")
        if not isinstance(self.faiss_metric_type, str) or not self.faiss_metric_type:
            raise TypeError(f"faiss_metric_type must be non-empty string; {self.faiss_metric_type!r} is invalid")
        if not isinstance(getattr(faiss, self.faiss_metric_type, None), int):
            raise ValueError(f"faiss_metric_type must specify a valid Faiss metric type; {self.faiss_metric_type!r} is invalid")

    @property
    def item_index_dir(self):
        if not hasattr(self, '_item_index_dir'):
            from .url_utils import use_s3
            model_in_path = self.agent.model_in_path
            self._item_index_dir = use_s3('%s%s/item_index/' % (model_in_path, self.index_type))
        return self._item_index_dir

    @property
    def item_index_partition_file_name(self):
        if not hasattr(self, '_item_index_partition_file_name'):
            rank = self.agent.rank
            worker_count = self.agent.worker_count
            self._item_index_partition_file_name = 'part_%d_%d.dat' % (worker_count, rank)
        return self._item_index_partition_file_name

    @property
    def item_index_partition_path(self):
        if not hasattr(self, '_item_index_partition_path'):
            dir_path = self.item_index_dir
            file_name = self.item_index_partition_file_name
            self._item_index_partition_path = '%s%s' % (dir_path, file_name)
        return self._item_index_partition_path

    def _make_index_meta(self):
        meta = super()._make_index_meta()
        meta['faiss_index_description'] = self.faiss_index_description
        meta['faiss_metric_type'] = self.faiss_metric_type
        return meta

    def _get_index_pattition_path(self, item_index_dir, partition_count, rank):
        partition_path = '%spart_%d_%d.dat' % (item_index_dir, partition_count, rank)
        return partition_path

    def _create_faiss_index_partition(self):
        item_embedding_size = self.agent.item_embedding_size
        faiss_index_description = self.faiss_index_description
        faiss_metric_type = self.faiss_metric_type
        params = item_embedding_size, faiss_index_description, faiss_metric_type
        print('faiss index params: %r' % (params,))
        metric_type = getattr(faiss, faiss_metric_type)
        params = item_embedding_size, faiss_index_description, metric_type
        faiss_index = faiss.index_factory(*params)
        self._faiss_index = faiss.IndexIDMap(faiss_index)

    def _output_faiss_index_partition(self):
        print('faiss index ntotal [%d]: %d' % (self.agent.rank, self._faiss_index.ntotal))
        _metaspore.ensure_local_directory(self.item_index_dir)
        item_index_stream = _metaspore.OutputStream(self.item_index_partition_path)
        item_index_writer = faiss.PyCallbackIOWriter(item_index_stream.write)
        faiss.write_index(self._faiss_index, item_index_writer)

    def _convert_spark_item_id_column_type(self, data_type):
        # the faiss index only supports int64 item id
        _item_id_column_type = super()._convert_spark_item_id_column_type(data_type)
        return 'int64'

    def get_item_id_ndarray(self, minibatch):
        id_ndarray = super().get_item_id_ndarray(minibatch)
        if id_ndarray.dtype != numpy.int64:
            try:
                id_ndarray = id_ndarray.astype(numpy.int64)
            except ValueError as ex:
                message = "the faiss index only supports int64 item id, "
                message += "can not convert the item id ndarray to int64 numpy ndarray; "
                message += "consider setting enable_item_id_mapping=True"
                raise RuntimeError(message) from ex
        return id_ndarray

    def output_item_embedding_batch(self, minibatch, embeddings, id_ndarray):
        self._faiss_index.add_with_ids(embeddings, id_ndarray)

    def _load_faiss_index(self):
        item_index_dir = self.item_index_dir
        meta = self.get_index_meta()
        partition_count = meta['partition_count']
        item_embedding_size = meta['item_embedding_size']
        self._faiss_index = faiss.IndexShards(item_embedding_size, True, False)
        for rank in range(partition_count):
            index_partition_path = self._get_index_pattition_path(item_index_dir, partition_count, rank)
            item_index_stream = _metaspore.InputStream(index_partition_path)
            item_index_reader = faiss.PyCallbackIOReader(item_index_stream.read)
            index = faiss.read_index(item_index_reader)
            self._faiss_index.add_shard(index)
            item_index_stream = None
        print('faiss index ntotal: %d' % self._faiss_index.ntotal)

    def search_item_embedding_batch(self, embeddings):
        k = self.agent.retrieval_item_count
        output_user_embeddings = self.agent.output_user_embeddings
        distances, indices = self._faiss_index.search(embeddings, k)
        indices = list(indices)
        distances = list(distances)
        embeddings = list(embeddings) if output_user_embeddings else None
        return indices, distances, embeddings

    def _unload_faiss_index(self):
        del self._faiss_index

    def begin_creating_index_partition(self):
        super().begin_creating_index_partition()
        self._create_faiss_index_partition()

    def end_creating_index_partition(self):
        super().end_creating_index_partition()
        self._output_faiss_index_partition()

    def begin_querying_index(self):
        super().begin_querying_index()
        if self.agent.is_worker:
            self._load_faiss_index()

    def end_querying_index(self):
        if self.agent.is_worker:
            self._unload_faiss_index()
        super().end_querying_index()

class TwoTowerMilvusIndexBuilder(TwoTowerIndexBuilder):
    def __init__(self, agent):
        super().__init__(agent)
        self.milvus_collection_name = getattr(agent, 'milvus_collection_name', None)
        self.milvus_host = getattr(agent, 'milvus_host', 'localhost')
        self.milvus_port = getattr(agent, 'milvus_port', 19530)
        self.milvus_item_id_field_name = getattr(agent, 'milvus_item_id_field_name', 'item_id')
        self.milvus_item_embedding_field_name = getattr(agent, 'milvus_item_embedding_field_name', 'item_embedding')
        self.milvus_string_item_id_max_length = getattr(agent, 'milvus_string_item_id_max_length', 65535)
        self.milvus_index_type = getattr(agent, 'milvus_index_type', 'IVF_FLAT')
        self.milvus_index_params = getattr(agent, 'milvus_index_params', {'nlist': 1024})
        self.milvus_metric_type = getattr(agent, 'milvus_metric_type', 'IP')
        self.milvus_search_params = getattr(agent, 'milvus_search_params', {'nprobe': 128})
        self.milvus_extra_fields = getattr(agent, 'milvus_extra_fields', None)
        self.milvus_extra_string_max_length = getattr(agent, 'milvus_extra_string_max_length', 65535)
        self.milvus_extra_array_multivalue_delimiter = getattr(agent, 'milvus_extra_array_multivalue_delimiter', '\001')
        if self.milvus_collection_name is None:
            raise RuntimeError("milvus_collection_name is required")
        if not isinstance(self.milvus_collection_name, str) or not self.milvus_collection_name:
            raise TypeError(f"milvus_collection_name must be non-empty string; {self.milvus_collection_name!r} is invalid")
        if not isinstance(self.milvus_host, str) or not self.milvus_host:
            raise TypeError(f"milvus_host must be non-empty string; {self.milvus_host!r} is invalid")
        if not isinstance(self.milvus_port, int) or self.milvus_port <= 0:
            raise TypeError(f"milvus_port must be positive integer; {self.milvus_port!r} is invalid")
        if not isinstance(self.milvus_item_id_field_name, str) or not self.milvus_item_id_field_name:
            raise TypeError(f"milvus_item_id_field_name must be non-empty string; {self.milvus_item_id_field_name!r} is invalid")
        if not isinstance(self.milvus_item_embedding_field_name, str) or not self.milvus_item_embedding_field_name:
            raise TypeError(f"milvus_item_embedding_field_name must be non-empty string; {self.milvus_item_embedding_field_name!r} is invalid")
        if not isinstance(self.milvus_string_item_id_max_length, int) or not 1 <= self.milvus_string_item_id_max_length <= 65535:
            raise TypeError(f"milvus_string_item_id_max_length must be integer between 1 and 65535; {self.milvus_string_item_id_max_length!r} is invalid")
        if not isinstance(self.milvus_index_type, str) or not self.milvus_index_type:
            raise TypeError(f"milvus_index_type must be non-empty string; {self.milvus_index_type!r} is invalid")
        if not isinstance(self.milvus_index_params, dict):
            raise TypeError(f"milvus_index_params must be dict; {self.milvus_index_params!r} is invalid")
        if not isinstance(self.milvus_metric_type, str) or not self.milvus_metric_type:
            raise TypeError(f"milvus_metric_type must be non-empty string; {self.milvus_metric_type!r} is invalid")
        if not isinstance(self.milvus_search_params, dict):
            raise TypeError(f"milvus_search_params must be dict; {self.milvus_search_params!r} is invalid")
        if self.milvus_extra_fields is not None and (
            not isinstance(self.milvus_extra_fields, (list, tuple)) or
            not all(isinstance(x, str) for x in self.milvus_extra_fields)):
            raise TypeError(f"milvus_extra_fields must be list or tuple of strings; "
                            f"{self.milvus_extra_fields!r} is invalid")
        if not isinstance(self.milvus_extra_string_max_length, int) or not 1 <= self.milvus_extra_string_max_length <= 65535:
            raise TypeError(f"milvus_extra_string_max_length must be integer between 1 and 65535; {self.milvus_extra_string_max_length!r} is invalid")
        if not isinstance(self.milvus_extra_array_multivalue_delimiter, str) or not self.milvus_extra_array_multivalue_delimiter:
            raise TypeError(f"milvus_extra_array_multivalue_delimiter must be non-empty string; {self.milvus_extra_array_multivalue_delimiter!r} is invalid")

    @staticmethod
    def _get_milvus_attributes():
        milvus_attributes = (
            'milvus_collection_name',
            'milvus_host',
            'milvus_port',
            'milvus_item_id_field_name',
            'milvus_item_embedding_field_name',
            'milvus_string_item_id_max_length',
            'milvus_index_type',
            'milvus_index_params',
            'milvus_metric_type',
            'milvus_search_params',
            'milvus_extra_fields',
            'milvus_extra_string_max_length',
            'milvus_extra_array_multivalue_delimiter',
        )
        return milvus_attributes

    @property
    def milvus_alias(self):
        if not hasattr(self, '_milvus_alias'):
            milvus_collection_name = self.milvus_collection_name
            if self.agent.is_coordinator:
                self._milvus_alias = '%s_connection_coordinator' % milvus_collection_name
            else:
                rank = self.agent.rank
                worker_count = self.agent.worker_count
                self._milvus_alias = '%s_connection_worker_%d_%d' % (milvus_collection_name, worker_count, rank)
        return self._milvus_alias

    def _make_index_meta(self):
        meta = super()._make_index_meta()
        meta['milvus_collection_name'] = self.milvus_collection_name
        meta['milvus_host'] = self.milvus_host
        meta['milvus_port'] = self.milvus_port
        meta['milvus_item_id_field_name'] = self.milvus_item_id_field_name
        meta['milvus_item_embedding_field_name'] = self.milvus_item_embedding_field_name
        meta['milvus_string_item_id_max_length'] = self.milvus_string_item_id_max_length
        meta['milvus_index_type'] = self.milvus_index_type
        meta['milvus_index_params'] = self.milvus_index_params
        meta['milvus_metric_type'] = self.milvus_metric_type
        meta['milvus_search_params'] = self.milvus_search_params
        meta['milvus_extra_fields'] = self.milvus_extra_fields
        meta['milvus_extra_string_max_length'] = self.milvus_extra_string_max_length
        meta['milvus_extra_array_multivalue_delimiter'] = self.milvus_extra_array_multivalue_delimiter
        return meta

    def _open_milvus_connection(self):
        from pymilvus import connections
        host = self.milvus_host
        port = self.milvus_port
        print("Open milvus connection %s" % self.milvus_alias)
        connections.connect(alias=self.milvus_alias, host=host, port=str(port))

    def _close_milvus_connection(self):
        from pymilvus import connections
        print("Close milvus connection %s" % self.milvus_alias)
        connections.disconnect(alias=self.milvus_alias)

    def _drop_milvus_collection_if_exists(self):
        from pymilvus import Collection
        from pymilvus import utility
        milvus_alias = self.milvus_alias
        milvus_collection_name = self.milvus_collection_name
        if utility.has_collection(collection_name=milvus_collection_name, using=milvus_alias):
            print("Drop existing milvus collection %s" % milvus_collection_name)
            collection = Collection(name=milvus_collection_name, using=milvus_alias)
            collection.drop()

    def _create_milvus_collection(self):
        from pymilvus import Collection
        milvus_alias = self.milvus_alias
        milvus_collection_name = self.milvus_collection_name
        milvus_schema = self._get_milvus_schema()
        print("Create milvus collection %s" % milvus_collection_name)
        self._milvus_collection = Collection(name=milvus_collection_name, schema=milvus_schema, using=milvus_alias)

    def _open_milvus_collection(self):
        from pymilvus import Collection
        milvus_alias = self.milvus_alias
        milvus_collection_name = self.milvus_collection_name
        print("Open milvus collection %s" % milvus_collection_name)
        self._milvus_collection = Collection(name=milvus_collection_name, using=milvus_alias)

    def _load_milvus_collection(self):
        from pymilvus import Collection
        milvus_alias = self.milvus_alias
        milvus_collection_name = self.milvus_collection_name
        print("Load milvus collection %s" % milvus_collection_name)
        self._milvus_collection = Collection(name=milvus_collection_name, using=milvus_alias)
        self._milvus_collection.load()

    def _release_milvus_collection(self):
        milvus_collection_name = self.milvus_collection_name
        print("Release milvus collection %s" % milvus_collection_name)
        self._milvus_collection.load()

    def _get_milvus_schema(self):
        from pymilvus import CollectionSchema
        milvus_fields = self._get_milvus_fields()
        milvus_collection_name = self.milvus_collection_name
        milvus_schema = CollectionSchema(fields=milvus_fields, description=milvus_collection_name)
        return milvus_schema

    def _get_milvus_fields(self):
        from pymilvus import DataType
        from pymilvus import FieldSchema
        item_id_field_name = self.milvus_item_id_field_name
        item_embedding_field_name = self.milvus_item_embedding_field_name
        item_embedding_size = self.agent.item_embedding_size
        milvus_fields = []
        if self.item_id_column_type == 'int64':
            milvus_fields.append(FieldSchema(name=item_id_field_name, dtype=DataType.INT64, is_primary=True))
        else:
            milvus_fields.append(FieldSchema(name=item_id_field_name, dtype=DataType.VARCHAR, is_primary=True,
                                             max_length=self.milvus_string_item_id_max_length))
        milvus_fields.append(FieldSchema(name=item_embedding_field_name, dtype=DataType.FLOAT_VECTOR,
                                         dim=item_embedding_size))
        self._add_milvus_extra_fields(milvus_fields)
        return milvus_fields

    def _add_milvus_extra_fields(self, milvus_fields):
        from .schema_utils import is_data_type_supported
        if not self.milvus_extra_fields:
            return
        item_dataset = self.agent.dataset
        for field_name in self.milvus_extra_fields:
            try:
                field = item_dataset.schema[field_name]
            except KeyError as ex:
                message = f"field_name {field_name!r} in milvus_extra_fields is not found "
                message += f"in the columns of item_dataset: {item_dataset.columns!r}"
                raise RuntimeError(message) from ex
            if not is_data_type_supported(field.dataType):
                message = "data type of column %r is not supported" % field.name
                raise RuntimeError(message)
            milvus_field = self._map_spark_field(field)
            milvus_fields.append(milvus_field)

    def _map_spark_field(self, field):
        from pymilvus import DataType
        from pymilvus import FieldSchema
        from pyspark.sql.types import StringType
        from pyspark.sql.types import FloatType
        from pyspark.sql.types import DoubleType
        from pyspark.sql.types import IntegerType
        from pyspark.sql.types import LongType
        from pyspark.sql.types import BooleanType
        from pyspark.sql.types import ArrayType
        name = field.name
        data_type = field.dataType
        max_length = self.milvus_extra_string_max_length
        types = (StringType, FloatType, DoubleType, IntegerType, LongType, BooleanType)
        if isinstance(data_type, types):
            if isinstance(data_type, StringType):
                return FieldSchema(name=name, dtype=DataType.VARCHAR, max_length=max_length)
            elif isinstance(data_type, FloatType):
                return FieldSchema(name=name, dtype=DataType.FLOAT)
            elif isinstance(data_type, DoubleType):
                return FieldSchema(name=name, dtype=DataType.DOUBLE)
            elif isinstance(data_type, IntegerType):
                return FieldSchema(name=name, dtype=DataType.INT32)
            elif isinstance(data_type, LongType):
                return FieldSchema(name=name, dtype=DataType.INT64)
            else:
                assert isinstance(data_type, BooleanType)
                return FieldSchema(name=name, dtype=DataType.BOOL)
        else:
            assert isinstance(data_type, ArrayType) and isinstance(data_type.elementType, types)
            return FieldSchema(name=name, dtype=DataType.VARCHAR, max_length=max_length)

    def _make_minibatch_processor(self):
        from .schema_utils import is_data_type_supported
        if not self.milvus_extra_fields:
            return ()
        item_dataset = self.agent.dataset
        minibatch_processor = []
        for field_name in self.milvus_extra_fields:
            try:
                field = item_dataset.schema[field_name]
            except KeyError as ex:
                message = f"field_name {field_name!r} in milvus_extra_fields is not found "
                message += f"in the columns of item_dataset: {item_dataset.columns!r}"
                raise RuntimeError(message) from ex
            if not is_data_type_supported(field.dataType):
                message = "data type of column %r is not supported" % field.name
                raise RuntimeError(message)
            column_processor = self._make_column_processor(field)
            minibatch_processor.append(column_processor)
        return tuple(minibatch_processor)

    def _make_column_processor(self, field):
        from pyspark.sql.types import StringType
        from pyspark.sql.types import FloatType
        from pyspark.sql.types import DoubleType
        from pyspark.sql.types import IntegerType
        from pyspark.sql.types import LongType
        from pyspark.sql.types import BooleanType
        from pyspark.sql.types import ArrayType
        name = field.name
        data_type = field.dataType
        delimiter = self.milvus_extra_array_multivalue_delimiter
        types = (StringType, FloatType, DoubleType, IntegerType, LongType, BooleanType)
        if isinstance(data_type, types):
            return lambda minibatch: minibatch[name].values
        else:
            assert isinstance(data_type, ArrayType) and isinstance(data_type.elementType, types)
            if isinstance(data_type.elementType, StringType):
                return lambda minibatch: numpy.array([delimiter.join(arr) for arr in minibatch[name]], dtype=object)
            elif isinstance(data_type.elementType, BooleanType):
                return lambda minibatch: numpy.array([delimiter.join(map(str, arr)).lower() for arr in minibatch[name]], dtype=object)
            else:
                return lambda minibatch: numpy.array([delimiter.join(map(str, arr)) for arr in minibatch[name]], dtype=object)

    def distribute_minibatch_processor(self):
        self.minibatch_processor = self._make_minibatch_processor()
        processor = self.minibatch_processor
        sc = self.agent.spark_context
        worker_count = self.agent.worker_count
        rdd = sc.parallelize(range(worker_count), worker_count)
        rdd.barrier().mapPartitions(lambda _: __class__._distribute_minibatch_processor(processor, _)).collect()

    @classmethod
    def _distribute_minibatch_processor(cls, processor, _):
        from .agent import Agent
        self = Agent.get_instance().index_builder
        self.minibatch_processor = processor
        return _

    def _create_milvus_index(self):
        milvus_collection_name = self.milvus_collection_name
        milvus_index_type = self.milvus_index_type
        milvus_index_params = self.milvus_index_params
        milvus_metric_type = self.milvus_metric_type
        item_embedding_field_name = self.milvus_item_embedding_field_name
        index_params = {"index_type": milvus_index_type,
                        "metric_type": milvus_metric_type,
                        "params": milvus_index_params}
        print("Creating milvus index %s" % milvus_collection_name)
        self._milvus_collection.create_index(field_name=item_embedding_field_name, index_params=index_params)

    def output_item_embedding_batch(self, minibatch, embeddings, id_ndarray):
        ndarrays = [id_ndarray, embeddings]
        self._add_extra_numpy_ndarrays(minibatch, ndarrays)
        self._milvus_collection.insert(ndarrays)

    def _add_extra_numpy_ndarrays(self, minibatch, ndarrays):
        processor = self.minibatch_processor
        extra_ndarrays = [proc(minibatch) for proc in processor]
        ndarrays += extra_ndarrays

    def search_item_embedding_batch(self, embeddings):
        k = self.agent.retrieval_item_count
        output_user_embeddings = self.agent.output_user_embeddings
        milvus_metric_type = self.milvus_metric_type
        milvus_search_params = self.milvus_search_params
        item_embedding_field_name = self.milvus_item_embedding_field_name
        search_params = {"metric_type": milvus_metric_type, "params": milvus_search_params}
        search_results = self._milvus_collection.search(
            data=embeddings,
            anns_field=item_embedding_field_name,
            param=search_params,
            limit=k,
            expr=None
        )
        indices = [result.ids for result in search_results]
        distances = [result.distances for result in search_results]
        embeddings = list(embeddings) if output_user_embeddings else None
        return indices, distances, embeddings

    def begin_creating_index(self):
        super().begin_creating_index()
        self._open_milvus_connection()
        self._drop_milvus_collection_if_exists()
        self._create_milvus_collection()

    def end_creating_index(self):
        self._create_milvus_index()
        self._close_milvus_connection()
        super().end_creating_index()

    def begin_creating_index_partition(self):
        super().begin_creating_index_partition()
        self._open_milvus_connection()
        self._open_milvus_collection()

    def end_creating_index_partition(self):
        self._close_milvus_connection()
        super().end_creating_index_partition()

    def distribute_state(self):
        super().distribute_state()
        self.distribute_minibatch_processor()

    def begin_querying_index(self):
        super().begin_querying_index()
        self._open_milvus_connection()
        if self.agent.is_coordinator:
            self._load_milvus_collection()
        else:
            self._open_milvus_collection()

    def end_querying_index(self):
        if self.agent.is_coordinator:
            self._release_milvus_collection()
        self._close_milvus_connection()
        super().end_querying_index()

class TwoTowerIndexBaseAgent(PyTorchAgent):
    def _get_metric_class(self):
        return ModelMetric

    def _get_index_builder_class(self):
        if self.index_builder_class is not None:
            return self.index_builder_class
        attributes = TwoTowerMilvusIndexBuilder._get_milvus_attributes()
        for attribute in attributes:
            if hasattr(self, attribute):
                return TwoTowerMilvusIndexBuilder
        return TwoTowerFaissIndexBuilder

    def _create_index_builder(self):
        index_builder_class = self._get_index_builder_class()
        index_builder = index_builder_class(self)
        return index_builder

    def distribute_index_builder_class(self):
        builder = self.index_builder_class
        rdd = self.spark_context.parallelize(range(self.worker_count), self.worker_count)
        rdd.barrier().mapPartitions(lambda _: __class__._distribute_index_builder_class(builder, _)).collect()

    @classmethod
    def _distribute_index_builder_class(cls, builder, _):
        self = __class__.get_instance()
        self.index_builder_class = builder
        return _

    def start_workers(self):
        self.distribute_index_builder_class()
        super().start_workers()

class TwoTowerIndexBuildingAgent(TwoTowerIndexBaseAgent):
    def start_workers(self):
        self.index_builder = self._create_index_builder()
        self.index_builder.begin_creating_index()
        super().start_workers()
        self.index_builder.distribute_state()

    def stop_workers(self):
        super().stop_workers()
        self.index_builder.end_creating_index()

    def worker_start(self):
        super().worker_start()
        self.index_builder = self._create_index_builder()
        self.index_builder.begin_creating_index_partition()

    def worker_stop(self):
        self.index_builder.end_creating_index_partition()
        super().worker_stop()

    def _default_feed_validation_dataset(self):
        import pyspark.sql.functions as F
        df = self.dataset
        if self.enable_item_id_mapping:
            df = df.withColumn(self.item_id_column_name, F.monotonically_increasing_id())
        func = self.feed_validation_minibatch()
        df = df.mapInPandas(func, df.schema)
        df.write.format('noop').mode('overwrite').save()

    def _default_preprocess_minibatch(self, minibatch):
        return minibatch

    def _default_validate_minibatch(self, minibatch):
        self.model.eval()
        minibatch = self.preprocess_minibatch(minibatch)
        predictions = self.model(minibatch)
        embeddings = predictions.detach().numpy()
        id_ndarray = self.index_builder.get_item_id_ndarray(minibatch)
        if self.index_builder.have_item_ids_partition_files:
            ids_data = self.index_builder.make_item_ids_mapping_batch(minibatch, embeddings, id_ndarray)
            self.index_builder.output_item_ids_mapping_batch(ids_data)
        self.index_builder.output_item_embedding_batch(minibatch, embeddings, id_ndarray)
        self.update_progress(batch_size=len(minibatch))
        return minibatch

class TwoTowerIndexRetrievalAgent(TwoTowerIndexBaseAgent):
    def start_workers(self):
        self.index_builder = self._create_index_builder()
        self.index_builder.begin_querying_index()
        super().start_workers()

    def stop_workers(self):
        super().stop_workers()
        self.index_builder.end_querying_index()

    def worker_start(self):
        super().worker_start()
        self.index_builder = self._create_index_builder()
        self.index_builder.begin_querying_index()

    def worker_stop(self):
        self.index_builder.end_querying_index()
        super().worker_stop()

    def _default_feed_validation_dataset(self):
        import pyspark.sql.functions as F
        self.item_ids_dataset = self.index_builder.load_item_ids()
        dataset = self.dataset.withColumn(self.increasing_id_column_name,
                                          F.monotonically_increasing_id())
        df = dataset
        func = self.feed_validation_minibatch()
        output_schema = self._make_validation_result_schema(df)
        df = df.mapInPandas(func, output_schema)
        df = self._rename_validation_result(df)
        self.dataset = dataset
        self.validation_result = df
        # PySpark DataFrame & RDD is lazily evaluated.
        # We must call ``cache`` here otherwise PySpark will try to reevaluate
        # ``validation_result`` when we use it, which is not possible as the
        # PS system has been shutdown.
        df.cache()
        df.write.format('noop').mode('overwrite').save()

    def _default_preprocess_minibatch(self, minibatch):
        return minibatch

    def _default_validate_minibatch(self, minibatch):
        self.model.eval()
        minibatch = self.preprocess_minibatch(minibatch)
        predictions = self.model(minibatch)
        embeddings = predictions.detach().numpy()
        indices, distances, embeddings = self.index_builder.search_item_embedding_batch(embeddings)
        minibatch = self._make_validation_result(minibatch, indices, distances, embeddings)
        self.update_progress(batch_size=len(minibatch))
        return minibatch

    def _make_validation_result(self, minibatch, indices, distances, embeddings):
        indices_name = self.recommendation_info_column_name + '_indices'
        distances_name = self.recommendation_info_column_name + '_distances'
        minibatch[indices_name] = indices
        minibatch[distances_name] = distances
        if embeddings is not None:
            user_embedding_name = self.recommendation_info_column_name + '_user_embedding'
            minibatch[user_embedding_name] = embeddings
        return minibatch

    def _make_validation_result_schema(self, df):
        from pyspark.sql.types import StructType
        from pyspark.sql.types import ArrayType
        from pyspark.sql.types import StringType
        from pyspark.sql.types import LongType
        from pyspark.sql.types import FloatType
        fields = []
        reserved = set()
        indices_name = self.recommendation_info_column_name + '_indices'
        distances_name = self.recommendation_info_column_name + '_distances'
        reserved.add(indices_name)
        reserved.add(distances_name)
        if self.output_user_embeddings:
            user_embedding_name = self.recommendation_info_column_name + '_user_embedding'
            reserved.add(user_embedding_name)
        for field in df.schema.fields:
            if field.name not in reserved:
                fields.append(field)
        meta = self.index_builder.get_index_meta()
        result_schema = StructType(fields)
        enable_item_id_mapping = meta['enable_item_id_mapping']
        item_id_column_type = meta['item_id_column_type']
        if enable_item_id_mapping:
            result_schema.add(indices_name, ArrayType(LongType()))
        else:
            id_column_type = LongType() if item_id_column_type == 'int64' else StringType()
            result_schema.add(indices_name, ArrayType(id_column_type))
        result_schema.add(distances_name, ArrayType(FloatType()))
        if self.output_user_embeddings:
            result_schema.add(user_embedding_name, ArrayType(FloatType()))
        return result_schema

    def _rename_validation_result(self, df):
        import pyspark.sql.functions as F
        indices_name = self.recommendation_info_column_name + '_indices'
        distances_name = self.recommendation_info_column_name + '_distances'
        indices = F.col(indices_name).alias('indices')
        distances = F.col(distances_name).alias('distances')
        if self.output_user_embeddings:
            user_embedding_name = self.recommendation_info_column_name + '_user_embedding'
            user_embedding = F.col(user_embedding_name).alias('user_embedding')
            df = df.select(self.increasing_id_column_name,
                           F.struct(indices, distances, user_embedding)
                            .alias(self.recommendation_info_column_name))
        else:
            df = df.select(self.increasing_id_column_name,
                           F.struct(indices, distances)
                            .alias(self.recommendation_info_column_name))
        return df

class TwoTowerRetrievalLauncher(PyTorchLauncher):
    def _initialize_agent(self, agent):
        agent.index_builder_class = self.index_builder_class
        super()._initialize_agent(agent)

class TwoTowerRetrievalHelperMixin(object):
    def __init__(self,
                 item_dataset=None,
                 index_builder_class=None,
                 index_building_agent_class=None,
                 retrieval_agent_class=None,
                 item_embedding_size=None,
                 item_id_column_name='item_id',
                 item_ids_column_indices=None,
                 item_ids_column_names=None,
                 item_ids_field_delimiter='\002',
                 item_ids_value_delimiter='\001',
                 enable_item_id_mapping=False,
                 output_item_embeddings=False,
                 output_user_embeddings=False,
                 increasing_id_column_name='iid',
                 recommendation_info_column_name='rec_info',
                 user_embedding_column_name='user_embedding',
                 retrieval_item_count=10,
                 **kwargs):
        super().__init__(**kwargs)
        self.item_dataset = item_dataset
        self.index_builder_class = index_builder_class
        self.index_building_agent_class = index_building_agent_class
        self.retrieval_agent_class = retrieval_agent_class
        self.item_embedding_size = item_embedding_size
        self.item_id_column_name = item_id_column_name
        self.item_ids_column_indices = item_ids_column_indices
        self.item_ids_column_names = item_ids_column_names
        self.item_ids_field_delimiter = item_ids_field_delimiter
        self.item_ids_value_delimiter = item_ids_value_delimiter
        self.enable_item_id_mapping = enable_item_id_mapping
        self.output_item_embeddings = output_item_embeddings
        self.output_user_embeddings = output_user_embeddings
        self.increasing_id_column_name = increasing_id_column_name
        self.recommendation_info_column_name = recommendation_info_column_name
        self.user_embedding_column_name = user_embedding_column_name
        self.retrieval_item_count = retrieval_item_count
        self.extra_agent_attributes['item_embedding_size'] = self.item_embedding_size
        self.extra_agent_attributes['item_id_column_name'] = self.item_id_column_name
        self.extra_agent_attributes['item_ids_column_indices'] = self.item_ids_column_indices
        self.extra_agent_attributes['item_ids_column_names'] = self.item_ids_column_names
        self.extra_agent_attributes['item_ids_field_delimiter'] = self.item_ids_field_delimiter
        self.extra_agent_attributes['item_ids_value_delimiter'] = self.item_ids_value_delimiter
        self.extra_agent_attributes['enable_item_id_mapping'] = self.enable_item_id_mapping
        self.extra_agent_attributes['output_item_embeddings'] = self.output_item_embeddings
        self.extra_agent_attributes['output_user_embeddings'] = self.output_user_embeddings
        self.extra_agent_attributes['increasing_id_column_name'] = self.increasing_id_column_name
        self.extra_agent_attributes['recommendation_info_column_name'] = self.recommendation_info_column_name
        self.extra_agent_attributes['user_embedding_column_name'] = self.user_embedding_column_name
        self.extra_agent_attributes['retrieval_item_count'] = self.retrieval_item_count

    def _check_properties(self):
        super()._check_properties()
        if not isinstance(self.module, TwoTowerRetrievalModule):
            raise TypeError(f"module must be TwoTowerRetrievalModule; {self.module!r} is invalid")
        if self.item_dataset is not None and not isinstance(self.item_dataset, pyspark.sql.DataFrame):
            raise TypeError(f"item_dataset must be pyspark.sql.DataFrame; {self.item_dataset!r} is invalid")
        if self.index_builder_class is not None and not issubclass(self.index_builder_class, TwoTowerIndexBuilder):
            raise TypeError(f"index_builder_class must be subclass of TwoTowerIndexBuilder; {self.index_builder_class!r} is invalid")
        if self.index_building_agent_class is not None and not issubclass(self.index_building_agent_class, TwoTowerIndexBuildingAgent):
            raise TypeError(f"index_building_agent_class must be subclass of TwoTowerIndexBuildingAgent; {self.index_building_agent_class!r} is invalid")
        if self.retrieval_agent_class is not None and not issubclass(self.retrieval_agent_class, TwoTowerIndexRetrievalAgent):
            raise TypeError(f"retrieval_agent_class must be subclass of TwoTowerIndexRetrievalAgent; {self.retrieval_agent_class!r} is invalid")
        if not isinstance(self.item_embedding_size, int) or self.item_embedding_size <= 0:
            raise TypeError(f"item_embedding_size must be positive integer; {self.item_embedding_size!r} is invalid")
        if not isinstance(self.item_id_column_name, str) or not self.item_id_column_name:
            raise TypeError(f"item_id_column_name must be non-empty string; {self.item_id_column_name!r} is invalid")
        if self.item_ids_column_indices is not None and (
           not isinstance(self.item_ids_column_indices, (list, tuple)) or not self.item_ids_column_indices or
           not all(isinstance(x, int) and x >= 0 for x in self.item_ids_column_indices)):
            raise TypeError(f"item_ids_column_indices must be non-empty list or tuple of non-negative integers; "
                            f"{self.item_ids_column_indices!r} is invalid")
        if self.item_ids_column_names is not None and (
           not isinstance(self.item_ids_column_names, (list, tuple)) or not self.item_ids_column_names or
           not all(isinstance(x, str) for x in self.item_ids_column_names)):
            raise TypeError(f"item_ids_column_names must be non-empty list or tuple of strings; "
                            f"{self.item_ids_column_names!r} is invalid")
        if not isinstance(self.item_ids_field_delimiter, str) or len(self.item_ids_field_delimiter) != 1:
            raise TypeError(f"item_ids_field_delimiter must be string of length 1; {self.item_ids_field_delimiter!r} is invalid")
        if not isinstance(self.item_ids_value_delimiter, str) or len(self.item_ids_value_delimiter) != 1:
            raise TypeError(f"item_ids_value_delimiter must be string of length 1; {self.item_ids_value_delimiter!r} is invalid")
        if not isinstance(self.increasing_id_column_name, str) or not self.increasing_id_column_name:
            raise TypeError(f"increasing_id_column_name must be non-empty string; {self.increasing_id_column_name!r} is invalid")
        if not isinstance(self.recommendation_info_column_name, str) or not self.recommendation_info_column_name:
            raise TypeError(f"recommendation_info_column_name must be non-empty string; {self.recommendation_info_column_name!r} is invalid")
        if not isinstance(self.user_embedding_column_name, str) or not self.user_embedding_column_name:
            raise TypeError(f"user_embedding_column_name must be non-empty string; {self.user_embedding_column_name!r} is invalid")
        if not isinstance(self.retrieval_item_count, int) or self.retrieval_item_count <= 0:
            raise TypeError(f"retrieval_item_count must be positive integer; {self.retrieval_item_count!r} is invalid")

    def _get_launcher_class(self):
        return TwoTowerRetrievalLauncher

    def _get_model_class(self):
        return TwoTowerRetrievalModel

    def _get_index_building_agent_class(self):
        return self.index_building_agent_class or TwoTowerIndexBuildingAgent

    def _get_retrieval_agent_class(self):
        return self.retrieval_agent_class or TwoTowerIndexRetrievalAgent

    def _create_launcher(self, dataset, is_training_mode):
        launcher = super()._create_launcher(dataset, is_training_mode)
        launcher.index_builder_class = self.index_builder_class
        return launcher

    def _get_model_arguments(self, module):
        args = super()._get_model_arguments(module)
        args['item_dataset'] = self.item_dataset
        args['index_builder_class'] = self.index_builder_class
        args['index_building_agent_class'] = self.index_building_agent_class
        args['retrieval_agent_class'] = self.retrieval_agent_class
        args['item_embedding_size'] = self.item_embedding_size
        args['item_id_column_name'] = self.item_id_column_name
        args['item_ids_column_indices'] = self.item_ids_column_indices
        args['item_ids_column_names'] = self.item_ids_column_names
        args['item_ids_field_delimiter'] = self.item_ids_field_delimiter
        args['item_ids_value_delimiter'] = self.item_ids_value_delimiter
        args['enable_item_id_mapping'] = self.enable_item_id_mapping
        args['output_item_embeddings'] = self.output_item_embeddings
        args['output_user_embeddings'] = self.output_user_embeddings
        args['increasing_id_column_name'] = self.increasing_id_column_name
        args['recommendation_info_column_name'] = self.recommendation_info_column_name
        args['user_embedding_column_name'] = self.user_embedding_column_name
        args['retrieval_item_count'] = self.retrieval_item_count
        return args

class TwoTowerRetrievalModel(TwoTowerRetrievalHelperMixin, PyTorchModel):
    def _get_item_struct_fields(self, item_ids_dataset):
        import pyspark.sql.functions as F
        fields = []
        if item_ids_dataset is None:
            fields.append(F.col('index').alias('name'))
        elif 'name' in item_ids_dataset.columns:
            fields.append('name')
        else:
            fields.append(F.col('id').alias('name'))
        fields.append('distance')
        if item_ids_dataset is not None and 'item_embedding' in item_ids_dataset.columns:
            fields.append('item_embedding')
        return fields

    def _transform_rec_info(self, rec_info, item_ids_dataset):
        # ``_transform_rec_info`` transforms raw ``rec_info`` into more usable form
        # with the help of ``item_ids_dataset``.
        #
        # In raw ``rec_info``, ``indices`` hold the raw item indices returned from
        # the embedding retrieval engine (Faiss, Milvus or others). During the
        # transformation process, those raw item indices are mapped to more readable
        # form by ``item_ids_dataset``.
        #
        # ``item_ids_dataset`` can be None and it can have two or three columns.
        import pyspark.sql.functions as F
        if self.output_user_embeddings:
            user_embeddings = rec_info.select(self.increasing_id_column_name,
                                              self.recommendation_info_column_name + '.user_embedding')
        # zip, explode and rename
        rec_info = rec_info.withColumn(self.recommendation_info_column_name,
                                       F.arrays_zip(
                                           self.recommendation_info_column_name + '.indices',
                                           self.recommendation_info_column_name + '.distances'))
        rec_info = rec_info.select(self.increasing_id_column_name,
                                   (F.posexplode(self.recommendation_info_column_name)
                                     .alias('pos', self.recommendation_info_column_name)))
        rec_info = rec_info.select(self.increasing_id_column_name,
                                   'pos',
                                   F.col(self.recommendation_info_column_name + '.0').alias('index'),
                                   F.col(self.recommendation_info_column_name + '.1').alias('distance'))
        # join and reverse the explode process
        if item_ids_dataset is not None:
            rec_info = rec_info.join(item_ids_dataset, F.col('index') == F.col('id'))
        fields = self._get_item_struct_fields(item_ids_dataset)
        rec_info = rec_info.select(self.increasing_id_column_name, 'pos',
                                   F.struct(*fields).alias(self.recommendation_info_column_name))
        w = pyspark.sql.Window.partitionBy(self.increasing_id_column_name).orderBy('pos')
        rec_info = rec_info.select(self.increasing_id_column_name, 'pos',
                                   (F.collect_list(self.recommendation_info_column_name)
                                    .over(w).alias(self.recommendation_info_column_name)))
        rec_info = rec_info.groupBy(self.increasing_id_column_name).agg(
                                    F.max(self.recommendation_info_column_name)
                                     .alias(self.recommendation_info_column_name))
        if self.output_user_embeddings:
            rec_info = rec_info.join(user_embeddings, self.increasing_id_column_name)
        return rec_info

    def _join_rec_info(self, df, rec_info):
        df = df.join(rec_info, self.increasing_id_column_name)
        df = df.orderBy(self.increasing_id_column_name)
        df = df.drop(self.increasing_id_column_name)
        return df

    def _transform(self, dataset):
        launcher = self._create_launcher(dataset, False)
        launcher.module = self.module.user_module
        launcher.tensor_name_prefix = '_user_module.'
        launcher.agent_class = self._get_retrieval_agent_class()
        launcher.launch()
        df = launcher.agent_object.dataset
        rec_info = launcher.agent_object.validation_result
        item_ids_dataset = launcher.agent_object.item_ids_dataset
        rec_info = self._transform_rec_info(rec_info, item_ids_dataset)
        result = self._join_rec_info(df, rec_info)
        self.final_metric = launcher.agent_object._metric
        return result

    def stringify(self, result,
                  recommendation_info_item_delimiter="\001",
                  recommendation_info_field_delimiter="\004",
                  item_embedding_value_delimiter="\003",
                  user_embedding_value_delimiter="\003"):
        import pyspark.sql.functions as F
        from pyspark.sql.functions import pandas_udf
        @pandas_udf('string')
        def format_rec_info(rec_info):
            import pandas as pd
            output = []
            for record in rec_info:
                string = ''
                for item in record:
                    if string:
                        string += recommendation_info_item_delimiter
                    string += item['name']
                    string += recommendation_info_field_delimiter
                    string += str(item['distance'])
                    if 'item_embedding' in item:
                        string += recommendation_info_field_delimiter
                        string += item_embedding_value_delimiter.join(map(str, item['item_embedding']))
                output.append(string)
            return pd.Series(output)
        result = result.withColumn(self.recommendation_info_column_name,
                                   format_rec_info(self.recommendation_info_column_name))
        if self.output_user_embeddings:
            result = result.withColumn(self.user_embedding_column_name,
                                       F.array_join(F.col(self.user_embedding_column_name),
                                                    user_embedding_value_delimiter))
        return result

class TwoTowerRetrievalEstimator(TwoTowerRetrievalHelperMixin, PyTorchEstimator):
    def _check_properties(self):
        super()._check_properties()
        if self.model_export_path is not None and self.item_dataset is None:
            raise RuntimeError("item_dataset must be specified to export model")
        if self.item_ids_column_indices is None and self.item_ids_column_names is None:
            raise RuntimeError("one of item_ids_column_indices and item_ids_column_names must be specified")
        if not self.enable_item_id_mapping:
            if self.item_ids_column_indices is not None and len(self.item_ids_column_indices) != 1:
                raise RuntimeError(f"when enable_item_id_mapping is false, item_ids_column_indices must contain "
                                   f"exactly one column index, {self.item_ids_column_indices!r} is invalid")
            if self.item_ids_column_names is not None and len(self.item_ids_column_names) != 1:
                raise RuntimeError(f"when enable_item_id_mapping is false, item_ids_column_names must contain "
                                   f"exactly one column name, {self.item_ids_column_names!r} is invalid")

    def _fit(self, dataset):
        self._clear_output()
        launcher = self._create_launcher(dataset, True)
        launcher.model_export_selector = lambda m: m.user_module, '_user_module.'
        launcher.launch()
        module = launcher.agent_object.module
        module.eval()
        if self.item_dataset is not None:
            launcher2 = self._create_launcher(self.item_dataset, False)
            launcher2.module = module.item_module
            launcher2.tensor_name_prefix = '_item_module.'
            launcher2.agent_class = self._get_index_building_agent_class()
            launcher2.model_in_path = self.model_out_path
            launcher2.model_out_path = None
            launcher2.launch()
        model = self._create_model(module)
        self.final_metric = launcher.agent_object._metric
        return model
