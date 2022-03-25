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
import faiss
import pyspark
from . import _metaspore
from .embedding import EmbeddingOperator
from .estimator import PyTorchAgent
from .estimator import PyTorchModel
from .estimator import PyTorchEstimator

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

class FaissIndexBuildingAgent(PyTorchAgent):
    def worker_start(self):
        super().worker_start()
        self.setup_faiss_index()

    def setup_faiss_index(self):
        from .url_utils import use_s3
        self.item_index_output_dir = use_s3('%sfaiss/item_index/' % self.model_in_path)
        self.item_index_output_path = '%spart_%d_%d.dat' % (self.item_index_output_dir, self.worker_count, self.rank)
        self.item_ids_output_dir = use_s3('%sfaiss/item_ids/' % self.model_in_path)
        self.item_ids_output_path = '%spart_%d_%d.dat' % (self.item_ids_output_dir, self.worker_count, self.rank)
        self.index_meta_output_dir = use_s3('%sfaiss/' % self.model_in_path)
        self.index_meta_output_path = '%sindex_meta.json' % self.index_meta_output_dir
        _metaspore.ensure_local_directory(self.item_index_output_dir)
        _metaspore.ensure_local_directory(self.item_ids_output_dir)
        _metaspore.ensure_local_directory(self.index_meta_output_dir)
        params = self.item_embedding_size, self.faiss_index_description, self.faiss_metric_type
        print('faiss index params: %r' % (params,))
        metric_type = getattr(faiss, self.faiss_metric_type)
        params = self.item_embedding_size, self.faiss_index_description, metric_type
        self.faiss_index = faiss.index_factory(*params)
        self.faiss_index = faiss.IndexIDMap(self.faiss_index)
        self.item_ids_stream = _metaspore.OutputStream(self.item_ids_output_path)

    def feed_validation_dataset(self):
        import pyspark.sql.functions as F
        df = self.dataset.withColumn(self.item_id_column_name, F.monotonically_increasing_id())
        df = df.select(self.feed_validation_minibatch()(*df.columns).alias('validate'))
        df.write.format('noop').mode('overwrite').save()

    def preprocess_minibatch(self, minibatch):
        ndarrays = [col.values for col in minibatch]
        return ndarrays

    def validate_minibatch(self, minibatch):
        self.model.eval()
        ndarrays = self.preprocess_minibatch(minibatch)
        predictions = self.model(ndarrays[:-1])
        ids_data = ''
        id_ndarray = ndarrays[-1]
        embeddings = predictions.detach().numpy()
        for i in range(len(id_ndarray)):
            ids_data += str(id_ndarray[i])
            ids_data += self.item_ids_field_delimiter
            for j, index in enumerate(self.item_ids_column_indices):
                if j > 0:
                    ids_data += self.item_ids_value_delimiter
                field = ndarrays[index][i]
                if field is not None:
                    ids_data += str(field)
            if self.output_item_embeddings:
                ids_data += self.item_ids_field_delimiter
                for k, value in enumerate(embeddings[i]):
                    if k > 0:
                        ids_data += ','
                    ids_data += str(value)
            ids_data += '\n'
        ids_data = ids_data.encode('utf-8')
        self.item_ids_stream.write(ids_data)
        self.faiss_index.add_with_ids(embeddings, id_ndarray)

    def get_index_meta(self):
        meta_version = 1
        partition_count = self.worker_count
        item_ids_field_delimiter = self.item_ids_field_delimiter
        item_ids_value_delimiter = self.item_ids_value_delimiter
        output_item_embeddings = self.output_item_embeddings
        meta = {
            'meta_version' : meta_version,
            'partition_count' : partition_count,
            'item_ids_field_delimiter' : item_ids_field_delimiter,
            'item_ids_value_delimiter' : item_ids_value_delimiter,
            'output_item_embeddings' : output_item_embeddings,
        }
        return meta

    def output_index_meta(self):
        if self.rank == 0:
            meta = self.get_index_meta()
            string = json.dumps(meta, separators=(',', ': '), indent=4)
            data = (string + '\n').encode('utf-8')
            _metaspore.stream_write_all(self.index_meta_output_path, data)

    def output_faiss_index(self):
        print('faiss index ntotal [%d]: %d' % (self.rank, self.faiss_index.ntotal))
        item_index_stream = _metaspore.OutputStream(self.item_index_output_path)
        item_index_writer = faiss.PyCallbackIOWriter(item_index_stream.write)
        faiss.write_index(self.faiss_index, item_index_writer)
        self.output_index_meta()
        self.item_ids_stream = None

    def worker_stop(self):
        self.output_faiss_index()
        super().worker_stop()

class FaissIndexRetrievalAgent(PyTorchAgent):
    def worker_start(self):
        super().worker_start()
        self.load_faiss_index()

    def get_index_meta(self):
        from .url_utils import use_s3
        index_meta_input_dir = use_s3('%sfaiss/' % self.model_in_path)
        index_meta_input_path = '%sindex_meta.json' % index_meta_input_dir
        data = _metaspore.stream_read_all(index_meta_input_path)
        string = data.decode('utf-8')
        meta = json.loads(string)
        return meta

    def get_partition_count(self):
        meta = self.get_index_meta()
        partition_count = meta['partition_count']
        return partition_count

    def load_faiss_index(self):
        from .url_utils import use_s3
        item_index_input_dir = use_s3('%sfaiss/item_index/' % self.model_in_path)
        self.faiss_index = faiss.IndexShards(self.item_embedding_size, True, False)
        partition_count = self.get_partition_count()
        for rank in range(partition_count):
            item_index_input_path = '%spart_%d_%d.dat' % (item_index_input_dir, partition_count, rank)
            item_index_stream = _metaspore.InputStream(item_index_input_path)
            item_index_reader = faiss.PyCallbackIOReader(item_index_stream.read)
            index = faiss.read_index(item_index_reader)
            self.faiss_index.add_shard(index)
        print('faiss index ntotal: %d' % self.faiss_index.ntotal)

    def load_item_ids(self):
        from .input import read_s3_csv
        import pyspark.sql.functions as F
        item_ids_input_dir = '%sfaiss/item_ids/' % self.model_in_path
        item_ids_input_path = '%s*' % item_ids_input_dir
        df = read_s3_csv(self.spark_session, item_ids_input_path, delimiter=self.item_ids_field_delimiter)
        df = df.withColumnRenamed('_c0', 'id').withColumnRenamed('_c1', 'name')
        if self.output_item_embeddings:
            df = df.withColumnRenamed('_c2', 'item_embedding')
            df = df.withColumn('item_embedding', F.split(F.col('item_embedding'), ',').cast('array<float>'))
        return df

    def feed_validation_dataset(self):
        import pyspark.sql.functions as F
        self.item_ids_dataset = self.load_item_ids()
        dataset = self.dataset.withColumn(self.increasing_id_column_name,
                                          F.monotonically_increasing_id())
        df = dataset.select(self.increasing_id_column_name,
                            (self.feed_validation_minibatch()(*self.dataset.columns)
                             .alias(self.recommendation_info_column_name)))
        self.dataset = dataset
        self.validation_result = df
        # PySpark DataFrame & RDD is lazily evaluated.
        # We must call ``cache`` here otherwise PySpark will try to reevaluate
        # ``validation_result`` when we use it, which is not possible as the
        # PS system has been shutdown.
        df.cache()
        df.write.format('noop').mode('overwrite').save()

    def feed_validation_minibatch(self):
        from pyspark.sql.functions import pandas_udf
        signature = 'indices: array<long>, distances: array<float>'
        if self.output_user_embeddings:
            signature += ', user_embedding: array<float>'
        @pandas_udf(signature)
        def _feed_validation_minibatch(*minibatch):
            self = __class__.get_instance()
            result = self.validate_minibatch(minibatch)
            return result
        return _feed_validation_minibatch

    def preprocess_minibatch(self, minibatch):
        ndarrays = [col.values for col in minibatch]
        return ndarrays

    def validate_minibatch(self, minibatch):
        import pandas as pd
        self.model.eval()
        ndarrays = self.preprocess_minibatch(minibatch)
        predictions = self.model(ndarrays)
        embeddings = predictions.detach().numpy()
        distances, indices = self.faiss_index.search(embeddings, self.retrieval_item_count)
        data = {'indices': list(indices), 'distances': list(distances)}
        if self.output_user_embeddings:
            data['user_embedding'] = list(embeddings)
        minibatch_size = len(minibatch[0])
        index = pd.RangeIndex(minibatch_size)
        return pd.DataFrame(data=data, index=index)

class TwoTowerRetrievalHelperMixin(object):
    def __init__(self,
                 item_dataset=None,
                 index_building_agent_class=None,
                 retrieval_agent_class=None,
                 item_embedding_size=None,
                 faiss_index_description='Flat',
                 faiss_metric_type='METRIC_INNER_PRODUCT',
                 item_id_column_name='item_id',
                 item_ids_column_indices=None,
                 item_ids_field_delimiter='\002',
                 item_ids_value_delimiter='\001',
                 output_item_embeddings=False,
                 output_user_embeddings=False,
                 increasing_id_column_name='iid',
                 recommendation_info_column_name='rec_info',
                 user_embedding_column_name='user_embedding',
                 retrieval_item_count=10,
                 **kwargs):
        super().__init__(**kwargs)
        self.item_dataset = item_dataset
        self.index_building_agent_class = index_building_agent_class
        self.retrieval_agent_class = retrieval_agent_class
        self.item_embedding_size = item_embedding_size
        self.faiss_index_description = faiss_index_description
        self.faiss_metric_type = faiss_metric_type
        self.item_id_column_name = item_id_column_name
        self.item_ids_column_indices = item_ids_column_indices
        self.item_ids_field_delimiter = item_ids_field_delimiter
        self.item_ids_value_delimiter = item_ids_value_delimiter
        self.output_item_embeddings = output_item_embeddings
        self.output_user_embeddings = output_user_embeddings
        self.increasing_id_column_name = increasing_id_column_name
        self.recommendation_info_column_name = recommendation_info_column_name
        self.user_embedding_column_name = user_embedding_column_name
        self.retrieval_item_count = retrieval_item_count
        self.extra_agent_attributes['item_embedding_size'] = self.item_embedding_size
        self.extra_agent_attributes['faiss_index_description'] = self.faiss_index_description
        self.extra_agent_attributes['faiss_metric_type'] = self.faiss_metric_type
        self.extra_agent_attributes['item_id_column_name'] = self.item_id_column_name
        self.extra_agent_attributes['item_ids_column_indices'] = self.item_ids_column_indices
        self.extra_agent_attributes['item_ids_field_delimiter'] = self.item_ids_field_delimiter
        self.extra_agent_attributes['item_ids_value_delimiter'] = self.item_ids_value_delimiter
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
        if self.index_building_agent_class is not None and not issubclass(self.index_building_agent_class, FaissIndexBuildingAgent):
            raise TypeError(f"index_building_agent_class must be subclass of FaissIndexBuildingAgent; {self.index_building_agent_class!r} is invalid")
        if self.retrieval_agent_class is not None and not issubclass(self.retrieval_agent_class, FaissIndexRetrievalAgent):
            raise TypeError(f"retrieval_agent_class must be subclass of FaissIndexRetrievalAgent; {self.retrieval_agent_class!r} is invalid")
        if not isinstance(self.item_embedding_size, int) or self.item_embedding_size <= 0:
            raise TypeError(f"item_embedding_size must be positive integer; {self.item_embedding_size!r} is invalid")
        if not isinstance(self.faiss_index_description, str) or not self.faiss_index_description:
            raise TypeError(f"faiss_index_description must be non-empty string; {self.faiss_index_description!r} is invalid")
        if not isinstance(self.faiss_metric_type, str) or not self.faiss_metric_type:
            raise TypeError(f"faiss_metric_type must be non-empty string; {self.faiss_metric_type!r} is invalid")
        if not isinstance(getattr(faiss, self.faiss_metric_type, None), int):
            raise ValueError(f"faiss_metric_type must specify a valid Faiss metric type; {self.faiss_metric_type!r} is invalid")
        if not isinstance(self.item_id_column_name, str) or not self.item_id_column_name:
            raise TypeError(f"item_id_column_name must be non-empty string; {self.item_id_column_name!r} is invalid")
        if self.item_ids_column_indices is not None and (
           not isinstance(self.item_ids_column_indices, (list, tuple)) or
           not all(isinstance(x, int) and x >= 0 for x in self.item_ids_column_indices)):
            raise TypeError(f"item_ids_column_indices must be list or tuple of non-negative integers; "
                            f"{self.item_ids_column_indices!r} is invalid")
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

    def _get_model_class(self):
        return TwoTowerRetrievalModel

    def _get_index_building_agent_class(self):
        return self.index_building_agent_class or FaissIndexBuildingAgent

    def _get_retrieval_agent_class(self):
        return self.retrieval_agent_class or FaissIndexRetrievalAgent

    def _get_model_arguments(self, module):
        args = super()._get_model_arguments(module)
        args['item_dataset'] = self.item_dataset
        args['index_building_agent_class'] = self.index_building_agent_class
        args['retrieval_agent_class'] = self.retrieval_agent_class
        args['item_embedding_size'] = self.item_embedding_size
        args['faiss_index_description'] = self.faiss_index_description
        args['faiss_metric_type'] = self.faiss_metric_type
        args['item_id_column_name'] = self.item_id_column_name
        args['item_ids_column_indices'] = self.item_ids_column_indices
        args['item_ids_field_delimiter'] = self.item_ids_field_delimiter
        args['item_ids_value_delimiter'] = self.item_ids_value_delimiter
        args['output_item_embeddings'] = self.output_item_embeddings
        args['output_user_embeddings'] = self.output_user_embeddings
        args['increasing_id_column_name'] = self.increasing_id_column_name
        args['recommendation_info_column_name'] = self.recommendation_info_column_name
        args['user_embedding_column_name'] = self.user_embedding_column_name
        args['retrieval_item_count'] = self.retrieval_item_count
        return args

    def _reload_combine_schemas(self, module):
        for name, mod in module.named_modules():
            if isinstance(mod, EmbeddingOperator):
                if mod.has_alternative_column_name_file_path:
                    mod.reload_combine_schema(True)

class TwoTowerRetrievalModel(TwoTowerRetrievalHelperMixin, PyTorchModel):
    def _transform_rec_info(self, rec_info, item_ids_dataset):
        import pyspark.sql.functions as F
        if self.output_user_embeddings:
            user_embeddings = rec_info.select(self.increasing_id_column_name,
                                              self.recommendation_info_column_name + '.user_embedding')
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
        rec_info = rec_info.join(item_ids_dataset, F.col('index') == F.col('id'))
        w = pyspark.sql.Window.partitionBy(self.increasing_id_column_name).orderBy('pos')
        if self.output_item_embeddings:
            rec_info = rec_info.select(self.increasing_id_column_name, 'pos',
                                       (F.struct('name', 'distance', 'item_embedding')
                                         .alias(self.recommendation_info_column_name)))
        else:
            rec_info = rec_info.select(self.increasing_id_column_name, 'pos',
                                       (F.struct('name', 'distance')
                                         .alias(self.recommendation_info_column_name)))
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
        self._reload_combine_schemas(self.module)
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
        output_item_embeddings = self.output_item_embeddings
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
                    if output_item_embeddings:
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
        if not isinstance(self.item_ids_column_indices, (list, tuple)) or \
           not all(isinstance(x, int) and x >= 0 for x in self.item_ids_column_indices):
            raise TypeError(f"item_ids_column_indices must be list or tuple of non-negative integers; "
                            f"{self.item_ids_column_indices!r} is invalid")

    def _copy_faiss_index(self):
        if self.model_export_path is not None:
            from .url_utils import use_s3
            from .file_utils import copy_dir
            src_path = use_s3('%sfaiss/' % self.model_out_path)
            dst_path = use_s3('%s%s.ptm.msd/faiss/' % (self.model_export_path, self.experiment_name))
            copy_dir(src_path, dst_path)

    def _fit(self, dataset):
        self._clear_output()
        launcher = self._create_launcher(dataset, True)
        launcher.model_export_selector = lambda m: m.user_module, '_user_module.'
        launcher.launch()
        module = launcher.agent_object.module
        self._reload_combine_schemas(module)
        module.eval()
        if self.item_dataset is not None:
            launcher2 = self._create_launcher(self.item_dataset, False)
            launcher2.module = module.item_module
            launcher2.tensor_name_prefix = '_item_module.'
            launcher2.agent_class = self._get_index_building_agent_class()
            launcher2.model_in_path = self.model_out_path
            launcher2.model_out_path = None
            launcher2.launch()
            self._copy_faiss_index()
        model = self._create_model(module)
        self.final_metric = launcher.agent_object._metric
        return model
