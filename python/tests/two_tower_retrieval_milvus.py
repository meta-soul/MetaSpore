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
import yaml
from os import name
import torch
import pyspark
from metaspore import _metaspore
from metaspore.embedding import EmbeddingOperator
from metaspore.estimator import PyTorchAgent
from metaspore.estimator import PyTorchModel
from metaspore.estimator import PyTorchEstimator

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


class MilvusIndexBuildingAgent(PyTorchAgent):
    def start_workers(self):
        self.init_milvus()
        super().start_workers()
        
    def worker_start(self):
        super().worker_start()
        self.setup_item_index()
        self.load_milvus_index()

    def worker_stop(self):
        super().worker_stop()
        self.output_item_index()
        self.close_milvus_index()
        
    def init_milvus(self):        
        # open milvus connection here
        from pymilvus import connections, utility, FieldSchema, CollectionSchema, DataType, Collection
        self.milvus_alias = '%s_%d_%d'%(self.milvus_description, self.worker_count, self.rank)
        print(f"\nCreate milvus connection...%s"%self.milvus_alias)
        connections.connect(alias=self.milvus_alias, host=self.milvus_host, port=self.milvus_port)
        
        if utility.has_collection(collection_name=self.milvus_description, using=self.milvus_alias):
            print(f"\nDrop the existing milvus collection...%s"%self.milvus_description)
            collection = Collection(name=self.milvus_description, using=self.milvus_alias)
            collection.drop()

        self.milvus_fields = [
            FieldSchema(name=self.item_id_column_name, dtype=DataType.INT64, is_primary=True),
            FieldSchema(name=self.milvus_embedding_field, dtype=DataType.FLOAT_VECTOR, dim=self.item_embedding_size)
        ]

        print(f"\nCreate milvus collection...%s"%self.milvus_description)
        self.milvus_schema = CollectionSchema(fields=self.milvus_fields, description=self.milvus_description)
        self.milvus_collection = Collection(name=self.milvus_description, schema=self.milvus_schema, using=self.milvus_alias)

    def load_milvus_index(self):
        # open milvus connection here
        from pymilvus import connections, utility, FieldSchema, CollectionSchema, DataType, Collection
        self.milvus_alias = '%s_%d_%d'%(self.milvus_description, self.worker_count, self.rank)
        print(f"\nCreate milvus connection...%s"%self.milvus_alias)
        connections.connect(alias=self.milvus_alias, host=self.milvus_host, port=self.milvus_port)

        print(f"\nLoad milvus collection...on %s"%self.milvus_alias)
        self.milvus_collection = Collection(name=self.milvus_description, using=self.milvus_alias)
        if self.rank==0:
            self.milvus_collection.load()
    
    def setup_item_index(self):
        from metaspore.url_utils import use_s3
        self.item_ids_output_dir = use_s3('%smilvus/item_ids/' % self.model_in_path)
        self.item_ids_output_path = '%spart_%d_%d.dat' % (self.item_ids_output_dir, self.worker_count, self.rank)
        print(f"\nLoad milvus item ids index file: %s" % self.item_ids_output_path)
        _metaspore.ensure_local_directory(self.item_ids_output_dir)
        self.item_ids_stream = _metaspore.OutputStream(self.item_ids_output_path)

    def output_item_index(self):
        self.item_ids_stream = None
        
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
        id_array = ndarrays[-1]
        predictions = self.model(ndarrays[:-1])
        # insert result into milvus
        self.milvus_collection.insert([
            id_array.tolist(),
            predictions.detach().numpy().tolist()
        ])

        # write id mappings to s3 file stream
        ids_data = ''
        embeddings = predictions.detach().numpy()
        for i in range(len(id_array)):
            ids_data += str(id_array[i])
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
    
    def close_milvus_index(self):
        from pymilvus import connections, utility, Collection
        # create index
        if self.rank==0:
            index_params = {"index_type": self.milvus_index_type, \
                            "metric_type": self.milvus_metric_type, \
                            "params": {"nlist": self.milvus_nlist}}
            print(f"\nCreating milvus index...%s"%self.milvus_alias)
            self.milvus_collection.create_index(field_name=self.milvus_embedding_field, index_params=index_params)

        print(f"\nRelease milvus connection...%s"%self.milvus_alias)
        connections.disconnect(alias=self.milvus_alias)


class MilvusIndexRetrievalAgent(PyTorchAgent):
    def worker_start(self):
        super().worker_start()
        self.load_milvus_index()

    def worker_stop(self):
        self.close_milvus_index()
        super().worker_stop()
    
    def load_milvus_index(self):    
        from pymilvus import connections, Collection
        self.milvus_alias = '%s_%d_%d'%(self.milvus_description, self.worker_count, self.rank)
        print(f"\nCreate milvus connection...%s"%self.milvus_alias)
        connections.connect(alias=self.milvus_alias, host=self.milvus_host, port=self.milvus_port)
        
        print(f"\nLoad milvus collection...on %s"%self.milvus_alias)
        self.milvus_collection = Collection(name=self.milvus_description, using=self.milvus_alias)
        
        if self.rank==0:
            self.milvus_collection.load()

    def load_item_ids(self):
        from metaspore.input import read_s3_csv
        import pyspark.sql.functions as F
        item_ids_input_dir = '%smilvus/item_ids/' % self.model_in_path
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
        embeddings = predictions.detach().numpy().tolist()

        # search from milvus
        search_params = {"metric_type": self.milvus_metric_type, "params": {"nprobe": self.milvus_nprobe}}
        search_results = self.milvus_collection.search(
            data=embeddings, 
            anns_field=self.milvus_embedding_field, 
            param=search_params, 
            limit=self.retrieval_item_count, 
            expr=None
        )
        
        # fill data
        data = {'indices': list(map(lambda x: x.ids, search_results)), \
                'distances': list(map(lambda x: x.distances, search_results))}  
        if self.output_user_embeddings:
            data['user_embedding'] = list(embeddings)
        
        minibatch_size = len(minibatch[0])
        index = pd.RangeIndex(minibatch_size)
        return pd.DataFrame(data=data, index=index)

    def close_milvus_index(self):
        from pymilvus import connections
        print(f"\nRelease milvus collection...on %s", self.milvus_alias)
        connections.disconnect(alias=self.milvus_alias)


class TwoTowerRetrievalHelperMixin(object):
    def __init__(self,
                 item_dataset=None,
                 index_building_agent_class=None,
                 retrieval_agent_class=None,
                 item_embedding_size=None,
                 item_id_column_name='item_id',
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
        self.item_id_column_name = item_id_column_name
        self.item_ids_field_delimiter = item_ids_field_delimiter
        self.item_ids_value_delimiter = item_ids_value_delimiter
        self.output_item_embeddings = output_item_embeddings
        self.output_user_embeddings = output_user_embeddings
        self.increasing_id_column_name = increasing_id_column_name
        self.recommendation_info_column_name = recommendation_info_column_name
        self.user_embedding_column_name = user_embedding_column_name
        self.retrieval_item_count = retrieval_item_count
        self.extra_agent_attributes['item_embedding_size'] = self.item_embedding_size
        self.extra_agent_attributes['item_id_column_name'] = self.item_id_column_name
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
        if self.index_building_agent_class is not None and not issubclass(self.index_building_agent_class, MilvusIndexBuildingAgent):
            raise TypeError(f"index_building_agent_class must be subclass of MilvusIndexBuildingAgent; {self.index_building_agent_class!r} is invalid")
        if self.retrieval_agent_class is not None and not issubclass(self.retrieval_agent_class, MilvusIndexRetrievalAgent):
            raise TypeError(f"retrieval_agent_class must be subclass of MilvusIndexRetrievalAgent; {self.retrieval_agent_class!r} is invalid")
        if not isinstance(self.item_embedding_size, int) or self.item_embedding_size <= 0:
            raise TypeError(f"item_embedding_size must be positive integer; {self.item_embedding_size!r} is invalid")
        if not isinstance(self.item_id_column_name, str) or not self.item_id_column_name:
            raise TypeError(f"item_id_column_name must be non-empty string; {self.item_id_column_name!r} is invalid")
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
        return self.index_building_agent_class or MilvusIndexBuildingAgent

    def _get_retrieval_agent_class(self):
        return self.retrieval_agent_class or MilvusIndexRetrievalAgent

    def _get_model_arguments(self, module):
        args = super()._get_model_arguments(module)
        args['item_dataset'] = self.item_dataset
        args['index_building_agent_class'] = self.index_building_agent_class
        args['retrieval_agent_class'] = self.retrieval_agent_class
        args['item_embedding_size'] = self.item_embedding_size
        args['item_id_column_name'] = self.item_id_column_name
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
        model = self._create_model(module)
        self.final_metric = launcher.agent_object._metric
        return model
