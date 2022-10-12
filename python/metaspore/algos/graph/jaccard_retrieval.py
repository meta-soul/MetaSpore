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

from sqlalchemy import alias
import pyspark.ml.base
import pyspark.sql.functions as F
from pyspark.sql import Window
from pyspark.sql.functions import col
from pyspark.sql.types import LongType
from pyspark.ml.feature import CountVectorizer, MinHashLSH


class JaccardModel(pyspark.ml.base.Model):
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
        string += "t -> concat(t._1, '%s', t._2)" % self._format_delimiter(self.item_score_delimiter)
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


class JaccardEstimator(pyspark.ml.base.Estimator):
    def __init__(self,
                user_id_column_name=None,
                item_id_column_name=None,
                behavior_column_name=None,
                behavior_filter_value=None,
                max_recommendation_count=20,
                jaccard_distance_threshold=20,
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
        self.jaccard_distance_threshold=jaccard_distance_threshold
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
        model = JaccardModel(df=df,
                    key_column_name=self.key_column_name,
                    value_column_name=self.value_column_name,
                    item_score_delimiter=self.item_score_delimiter,
                    item_score_pair_delimiter=self.item_score_pair_delimiter,
                    item_id_column_name=self.item_id_column_name)
        
        return model
    
    ## compute the item similarity
    def _jaccard_transform(self, dataset):
        relationship_data = dataset.groupBy(F.col('item_id'))\
                                .agg(F.collect_list(F.col('user_id'))\
                                .alias('user_list'))
        ## 'user_list' column must be array<string> type
        cv = CountVectorizer(inputCol='user_list', outputCol='features')
        model_cv = cv.fit(relationship_data)
        cv_result = model_cv.transform(relationship_data)
        mh = MinHashLSH(inputCol='features', outputCol='hashes')
        model_mh = mh.fit(cv_result)
        jaccard_dist_table = model_mh.approxSimilarityJoin(cv_result, cv_result, threshold=self.jaccard_distance_threshold, distCol='jaccard_dist')\
                                                .select(F.col('datasetA.item_id').alias('item_id_i'),\
                                                        F.col('datasetB.item_id').alias('item_id_j'),\
                                                        F.col('jaccard_dist'))
        jaccard_sim_table = jaccard_dist_table.withColumn('jaccard_sim', 1-F.col('jaccard_dist')).drop('jaccard_dist')\
                                              .filter(F.col('jaccard_sim') != 0)\
                                              .filter(F.col('item_id_i') != F.col('item_id_j'))
        
        w = Window.partitionBy('item_id_i').orderBy(F.desc('jaccard_sim'))
        recall_result = jaccard_sim_table.withColumn('rn',F.row_number()\
                            .over(w))\
                            .filter(f'rn <= %d' % self.max_recommendation_count)\
                            .groupby('item_id_i')\
                            .agg(F.collect_list(F.struct(F.col('item_id_j').alias('_1'), F.col('jaccard_sim').alias('_2'))).alias(self.value_column_name))\
                            .withColumnRenamed('item_id_i', self.key_column_name)

        if self.debug:
            print('Debug --- jaccard result:')
            recall_result.show(10)
        return recall_result
    
    def _fit(self, dataset):
        dataset = self._filter_dataset(dataset)
        dataset = self._preprocess_dataset(dataset)
        df = self._jaccard_transform(dataset)
        model = self._create_model(df)
        return model
