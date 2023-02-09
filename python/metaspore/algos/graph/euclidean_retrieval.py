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

import pyspark.ml.base
import pyspark.sql.functions as F
from pyspark.sql import Window
from pyspark.sql.functions import col
from pyspark.sql.types import LongType
from pyspark.ml.feature import BucketedRandomProjectionLSH, CountVectorizer, MinMaxScaler, VectorAssembler
from pyspark.sql.types import FloatType

class EuclideanModel(pyspark.ml.base.Model):
    def __init__(self,
                df=None,
                key_column_name='key',
                value_column_name='value',
                item_score_delimiter=':',
                item_score_pair_delimiter=';',
                item_id_column_name=None,
                debug=False):
        super().__init__()
        self.df = df
        self.key_column_name = key_column_name
        self.value_column_name = value_column_name
        self.item_score_delimiter = item_score_delimiter
        self.item_score_pair_delimiter = item_score_pair_delimiter
        self.item_id_column_name = item_id_column_name
        self.debug = debug
        
    def _transform(self, dataset):
        if self.item_id_column_name is None:
            raise ValueError("item_id_column_name is required")
        on = dataset[self.item_id_column_name] == self.df[self.key_column_name]     
        return dataset.join(self.df, on=on, how='left_outer')
    
    def _format_delimiter(self, string):
        return ''.join('\\u%04X' % ord(c) for c in string)

    def _get_value_expr(self):
        string = "array_join(transform(%s, " % self.value_column_name
        string += "t -> concat(t.item_id, '%s', t.score)" % self._format_delimiter(self.item_score_delimiter)
        string += "), '%s') " % self._format_delimiter(self.item_score_pair_delimiter)
        string += "AS %s" % self.value_column_name
        return string

    def stringify(self):
        key = self.key_column_name
        value = self._get_value_expr()
        self.df = self.df.selectExpr(key, value)
        return self
    
    def publish(self):
        pass


class EuclideanEstimator(pyspark.ml.base.Estimator):
    def __init__(self,
                user_id_column_name=None,
                item_id_column_name=None,
                behavior_column_name=None,
                behavior_filter_value=None,
                max_recommendation_count=20,
                euclidean_distance_threshold=20,
                euclidean_bucket_length=20,
                key_column_name='key',
                value_column_name='value',
                item_score_delimiter=':',
                item_score_pair_delimiter=';',
                debug=False):
        super().__init__()
        self.user_id_column_name = user_id_column_name
        self.item_id_column_name = item_id_column_name
        self.behavior_column_name = behavior_column_name
        self.behavior_filter_value = behavior_filter_value
        self.max_recommendation_count = max_recommendation_count
        self.euclidean_distance_threshold=euclidean_distance_threshold
        self.euclidean_bucket_length=euclidean_bucket_length
        self.key_column_name = key_column_name
        self.value_column_name = value_column_name
        self.item_score_delimiter = item_score_delimiter
        self.item_score_pair_delimiter = item_score_pair_delimiter
        self.debug = debug
            
    def _filter_dataset(self, dataset):
        if self.behavior_column_name is None and self.behavior_filter_value is None:
            return dataset
        if self.behavior_column_name is not None and self.behavior_filter_value is not None:
            return dataset.where(dataset[self.behavior_column_name] == self.behavior_filter_value)
        
        raise RuntimeError("behavior_column_name and behavior_filter_value must be neither set or both set")
    
    def _preprocess_dataset(self, dataset):
        if self.user_id_column_name is None:
            raise ValueError("user_id_column_name is required")
        if self.item_id_column_name is None:
            raise ValueError("item_id_column_name is required")

        return dataset.select(col(self.user_id_column_name).alias("user_id"), 
                           col(self.item_id_column_name).alias("item_id"))
      
    def _create_model(self, df):
        model = EuclideanModel(df=df,
                    key_column_name=self.key_column_name,
                    value_column_name=self.value_column_name,
                    item_score_delimiter=self.item_score_delimiter,
                    item_score_pair_delimiter=self.item_score_pair_delimiter,
                    item_id_column_name=self.item_id_column_name)
        
        return model
    
    ## compute the item similarity
    def _euclidean_transform(self, dataset):
        relationship_data = dataset.groupBy(F.col('item_id'))\
                                .agg(F.collect_list(F.col('user_id'))\
                                .alias('user_list'))
        ## 'user_list' column must be array<string> type
        cv = CountVectorizer(inputCol='user_list', outputCol='features')
        model_cv = cv.fit(relationship_data)
        cv_result = model_cv.transform(relationship_data)
        mh = BucketedRandomProjectionLSH(inputCol='features', outputCol='hashes', bucketLength=self.euclidean_bucket_length)
        model_mh = mh.fit(cv_result)
        euclidean_dist_table = model_mh.approxSimilarityJoin(cv_result, cv_result, threshold=self.euclidean_distance_threshold, distCol='euclidean_dist')\
                                                .select(F.col('datasetA.item_id').alias('item_id_i'),\
                                                        F.col('datasetB.item_id').alias('item_id_j'),\
                                                        F.col('euclidean_dist'))
        vectorAssembler = VectorAssembler(handleInvalid="keep").setInputCols(['euclidean_dist']).setOutputCol('euclidean_dist_vec')
        euclidean_dist_table = vectorAssembler.transform(euclidean_dist_table)
        mmScaler = MinMaxScaler(outputCol="scaled_dist").setInputCol("euclidean_dist_vec")
        model = mmScaler.fit(euclidean_dist_table)
        euclidean_dist_table = model.transform(euclidean_dist_table)
        udf = F.udf(lambda x : float(x[0]), FloatType())
        euclidean_sim_table = euclidean_dist_table.withColumn('euclidean_sim', 1-udf('scaled_dist'))\
                                                  .drop('euclidean_dist', 'euclidean_dist_vec', 'scaled_dist')\
                                                  .filter(F.col('item_id_i') != F.col('item_id_j'))\

        w = Window.partitionBy('item_id_i').orderBy(F.desc('euclidean_sim'))
        recall_result = euclidean_sim_table.withColumn('rn',F.row_number()\
                            .over(w))\
                            .filter(f'rn <= %d' % self.max_recommendation_count)\
                            .groupby('item_id_i')\
                            .agg(F.collect_list(F.struct(F.col('item_id_j').alias('item_id'), F.col('euclidean_sim').alias('score'))).alias(self.value_column_name))\
                            .withColumnRenamed('item_id_i', self.key_column_name)
        
        if self.debug:
            print('Debug --- euclidean result:')
            recall_result.show(10)
        return recall_result
      
    def _fit(self, dataset):
        dataset = self._filter_dataset(dataset)
        dataset = self._preprocess_dataset(dataset)
        df = self._euclidean_transform(dataset)
        model = self._create_model(df)
        return model
