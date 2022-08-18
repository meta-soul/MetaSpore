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

from wsgiref.validate import validator
import metaspore as ms
import attrs
import cattrs
import logging

from logging import Logger
from typing import Dict, Tuple, Optional
from pyspark.sql import DataFrame, Window, functions as F 
from pyspark.ml.feature import CountVectorizer, BucketedRandomProjectionLSH, MinHashLSH, MinMaxScaler, VectorAssembler
from pyspark.mllib.evaluation import RankingMetrics
from pyspark.sql.types import FloatType

from ..constants import *

logger = logging.getLogger(__name__)

@attrs.frozen(kw_only=True)
class LSHRetrievalConfig:
    max_recommendation_count = attrs.field(validator=attrs.validators.instance_of(int))
    distance_metric = attrs.field(validator=attrs.validators.instance_of(str))
    distance_threshold = attrs.field(validator=attrs.validators.instance_of(float))
    lsh_bucket_length = attrs.field(default=None, 
        validator=attrs.validators.optional(attrs.validators.instance_of(int)))
    model_out_path = attrs.field(default=None, 
        validator=attrs.validators.optional(attrs.validators.instance_of(str)))

class LSHRetrievalModule:
    def __init__(self, conf: LSHRetrievalConfig):
        if isinstance(conf, dict):
            self.conf = LSHRetrievalModule.convert(conf)
        elif isinstance(conf, LSHRetrievalConfig):
            self.conf = conf
        else:
            raise TypeError("Type of 'conf' must be dict or LSHRetrievalConfig. Current type is {}".format(type(conf)))

    @staticmethod
    def convert(conf: dict) -> LSHRetrievalConfig:
        conf = cattrs.structure(conf['estimator_params'], LSHRetrievalConfig)
        return conf

    def train_euclidean(self, train_dataset, user_id_column, item_id_column, label_column, label_value, \
                        bucket_length, distance_threshold, max_recommendation_count):
        train_dataset = train_dataset.filter(F.col(label_column)==label_value)\
            .groupBy(item_id_column)\
            .agg(F.collect_list(user_id_column)\
            .alias('user_list'))
        ## 'user_list' column must be array<string> type
        cv = CountVectorizer(inputCol='user_list', outputCol='features')
        model_cv = cv.fit(train_dataset)
        cv_result = model_cv.transform(train_dataset)
        mh = BucketedRandomProjectionLSH(inputCol='features', outputCol='hashes', bucketLength=bucket_length)
        model_mh = mh.fit(cv_result)
        ## calculate the distance
        euclidean_dist_table = model_mh\
            .approxSimilarityJoin(cv_result, cv_result, threshold=distance_threshold, distCol='euclidean_dist')\
            .select(F.col('datasetA.' + item_id_column).alias('friend_A'),\
                    F.col('datasetB.' + item_id_column).alias('friend_B'),\
                    F.col('euclidean_dist'))
        vector_assembler = VectorAssembler(handleInvalid="keep")\
            .setInputCols(['euclidean_dist'])\
            .setOutputCol('euclidean_dist_vec')
        euclidean_dist_table = vector_assembler.transform(euclidean_dist_table)
        mm_scaler = MinMaxScaler(outputCol="scaled_dist").setInputCol("euclidean_dist_vec")
        model = mm_scaler.fit(euclidean_dist_table)
        euclidean_dist_table = model.transform(euclidean_dist_table)
        udf = F.udf(lambda x : float(x[0]), FloatType())
        euclidean_sim_table = euclidean_dist_table\
            .withColumn('euclidean_sim', 1-udf('scaled_dist'))\
            .drop('euclidean_dist', 'euclidean_dist_vec', 'scaled_dist')\
            .filter(F.col('friend_A') != F.col('friend_B'))\
            .withColumn('value', F.struct('friend_B', 'euclidean_sim'))
        w = Window.partitionBy('friend_A').orderBy(F.desc('euclidean_sim'))
        recall_result = euclidean_sim_table\
            .withColumn('rn', F.row_number().over(w))\
            .filter(f'rn <= %d' % max_recommendation_count)\
            .groupby('friend_A')\
            .agg(F.collect_list('value').alias('value_list'))\
            .withColumnRenamed('friend_A', 'key')
        return recall_result

    def train_jaccard(self, train_dataset, label_column, label_value, user_id_column, item_id_column, \
                      distance_threshold, max_recommendation_count):
        train_dataset = train_dataset.filter(F.col(label_column)==label_value)\
            .groupBy(item_id_column)\
            .agg(F.collect_list(user_id_column)\
            .alias('user_list'))
        ## 'user_list' column must be array<string> type
        cv = CountVectorizer(inputCol='user_list', outputCol='features')
        model_cv = cv.fit(train_dataset)
        cv_result = model_cv.transform(train_dataset)
        mh = MinHashLSH(inputCol='features', outputCol='hashes')
        model_mh = mh.fit(cv_result)
        jaccard_dist_table = model_mh\
            .approxSimilarityJoin(cv_result, cv_result, threshold=distance_threshold, distCol='jaccard_dist')\
            .select(F.col('datasetA.' + item_id_column).alias('friend_A'),\
                    F.col('datasetB.' + item_id_column).alias('friend_B'),\
                    F.col('jaccard_dist'))
        jaccard_sim_table = jaccard_dist_table\
            .withColumn('jaccard_sim', 1 - F.col('jaccard_dist')).drop('jaccard_dist')\
            .filter(F.col('jaccard_sim') != 0)\
            .filter(F.col('friend_A') != F.col('friend_B'))\
            .withColumn('value', F.struct('friend_B', 'jaccard_sim'))
        w = Window.partitionBy('friend_A').orderBy(F.desc('jaccard_sim'))
        recall_result = jaccard_sim_table.withColumn('rn',F.row_number()\
            .over(w))\
            .filter(f'rn <= %d' % max_recommendation_count)\
            .groupby('friend_A')\
            .agg(F.collect_list('value').alias('value_list'))\
            .withColumnRenamed('friend_A', 'key')
        return recall_result

    def transform(self, test_dataset, recall_result, item_id_column):
        cond = test_dataset[item_id_column]==recall_result['key']
        test_result = test_dataset.join(recall_result, on=cond, how='left')
        str_schema = 'array<struct<name:string,_2:double>>'
        test_result = test_result.withColumn('rec_info', F.col('value_list').cast(str_schema))
        return test_result

    def evaluate(elf, test_result, item_id_column):
        prediction_label_rdd = test_result.rdd.map(lambda x:(\
            [xx.name for xx in x.rec_info] if x.rec_info is not None else [], \
            [getattr(x, item_id_column)]))
        metrics = RankingMetrics(prediction_label_rdd)
        metric_dict = {}
        metric_dict['Precision@{}'.format(METRIC_RETRIEVAL_COUNT)] = metrics.precisionAt(METRIC_RETRIEVAL_COUNT)
        metric_dict['Recall@{}'.format(METRIC_RETRIEVAL_COUNT)] = metrics.recallAt(METRIC_RETRIEVAL_COUNT)
        metric_dict['MAP@{}'.format(METRIC_RETRIEVAL_COUNT)] = metrics.meanAveragePrecisionAt(METRIC_RETRIEVAL_COUNT)
        metric_dict['NDCG@{}'.format(METRIC_RETRIEVAL_COUNT)] = metrics.ndcgAt(METRIC_RETRIEVAL_COUNT)
        logger.info('Popular - evaluation: done')
        return metric_dict

    def run(self, train_dataset, test_dataset,  label_column='label', label_value='1',
            user_id_column='user_id', item_id_column='item_id'):
        if not isinstance(train_dataset, DataFrame):
            raise ValueError("Type of train_dataset must be DataFrame.")
        
        if self.conf.distance_metric == 'Jaccard':
            lsh_match = self.train_jaccard(train_dataset, label_column, label_value, user_id_column, item_id_column, 
                self.conf.distance_threshold, self.conf.max_recommendation_count)
        elif self.conf.distance_metric == 'Euclidean':
            lsh_match = self.train_euclidean(train_dataset, label_column, label_value, user_id_column, item_id_column, 
                self.conf.lsh_bucket_length, self.conf.distance_threshold, self.conf.max_recommendation_count)
        else:
            raise ValueError("Type of distance_metric must be in [Jaccard, Euclidean].")
        
        if self.conf.model_out_path:
            lsh_match.write.parquet(self.conf.model_out_path, mode="overwrite")
            logger.info('Popular - persistence: done')

        metric_dict = {}
        if test_dataset:
            if not isinstance(test_dataset, DataFrame):
                raise ValueError("Type of test_dataset must be DataFrame.") 
            test_result = self.transform(test_dataset, lsh_match, item_id_column)
            metric_dict = self.evaluate(test_result, item_id_column)
            logger.info('LSH evaluation metrics: {}'.format(metric_dict))

        return lsh_match, metric_dict
