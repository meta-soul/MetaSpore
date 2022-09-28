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


class ItemCFModel(pyspark.ml.base.Model):
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


class ItemCFEstimator(pyspark.ml.base.Estimator):
    def __init__(self,
                user_id_column_name=None,
                item_id_column_name=None,
                behavior_column_name=None,
                behavior_filter_value=None,
                max_recommendation_count=20,
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
        model = ItemCFModel(df=df,
                    key_column_name=self.key_column_name,
                    value_column_name=self.value_column_name,
                    item_score_delimiter=self.item_score_delimiter,
                    item_score_pair_delimiter=self.item_score_pair_delimiter,
                    item_id_column_name=self.item_id_column_name)

        return model

    ## compute w_u = 1 / sqrt{|I_u|}
    def _cf_compute_user_weight(self, dataset):
        user_weight = dataset.groupBy(F.col('user_id')) \
                        .agg(F.count(F.col('item_id')).alias('item_count')) \
                        .withColumn("item_count", F.col("item_count").cast(LongType())) \
                        .filter(F.col("user_id").isNotNull() & F.col('item_count').isNotNull() & (F.col('item_count')>0)) \
                        .withColumn('weight', F.lit(1)/F.sqrt(F.col('item_count')))

        return user_weight

    ## compute w_u for all users that click/purchase item_i and item_j
    def _cf_compute_crossing_weight(self, dataset, user_weight):
        t1 = dataset.withColumnRenamed('item_id', 'item_id_i')
        t2 = dataset.withColumnRenamed('item_id', 'item_id_j')

        crossing = t1.alias('t1').join(t2.alias('t2'), on=(F.col('t1.user_id')==F.col('t2.user_id')), how='leftouter')\
                        .filter(F.col('t1.user_id').isNotNull() & F.col('t2.user_id').isNotNull() \
                                                                & (F.col('t1.item_id_i')!=F.col('t2.item_id_j'))) \
                        .groupby('t1.user_id', 't1.item_id_i', 't2.item_id_j') \
                        .agg(F.count(F.lit(1)).alias('crossing_count'))

        crossing_weight = crossing.alias('t1').join(user_weight.alias('t2'), on=(F.col('t1.user_id')==F.col('t2.user_id'))) \
                        .filter(F.col('t2.item_count')>0) \
                        .select('t1.user_id', 't1.item_id_i', 't1.item_id_j', 't2.item_count', 't2.weight') \
                        .groupby('user_id', 'item_id_i', 'item_id_j') \
                        .agg(F.sum(F.col('weight')).alias('weight'))

        return crossing_weight

    ## compute l2_norm = \sqrt{\sum_{u \in U_i} w_u^2}
    def _cf_compute_item_l2_norm(self, dataset, user_weight):
        item_l2_norm = dataset.alias('t1').join(user_weight.alias('t2'), on=(F.col('t1.user_id')==F.col('t2.user_id'))) \
                            .filter(F.col('t2.item_count')>0) \
                            .select('t1.user_id', 't1.item_id', 't2.item_count', 't2.weight') \
                            .groupby('t1.item_id') \
                            .agg(F.sum(F.col('weight') * F.col('weight')).alias('weight')) \
                            .withColumn('weight', F.sqrt(F.col('weight')))

        return item_l2_norm

    def _cf_transform(self, dataset):
        user_weight = self._cf_compute_user_weight(dataset)
        if self.debug:
            print('Debug --- user bhv_count:')
            user_weight.show(10)

        crossing_weight = self._cf_compute_crossing_weight(dataset, user_weight)
        if self.debug:
            print('Debug --- crossing weight matrix:')
            crossing_weight.show(10)

        item_l2_norm = self._cf_compute_item_l2_norm(dataset, user_weight)
        if self.debug:
            print('Debug --- item l2 norm:')
            item_l2_norm.show(10)

        t2 = item_l2_norm.withColumnRenamed('weight', 'normal_weight_i')
        t3 = item_l2_norm.withColumnRenamed('weight', 'normal_weight_j')
        ## sparse inner product
        inner_product = crossing_weight.groupby('item_id_i', 'item_id_j') \
                                .agg(F.sum(F.col('weight') * F.col('weight')).alias('weight_sum'))
        ## penalized by the l2 norm
        cossine_similarity = inner_product.alias('t1')\
                                .join(t2.alias('t2'), on=(F.col('t1.item_id_i')==F.col('t2.item_id'))) \
                                .join(t3.alias('t3'), on=(F.col('t1.item_id_j')==F.col('t3.item_id'))) \
                                .withColumn('weight', F.col('t1.weight_sum')/(F.col('t2.normal_weight_i') * F.col('t3.normal_weight_j')))
        ## collect the top k list
        result = cossine_similarity.withColumn("rn", F.row_number().over(Window.partitionBy('item_id_i').orderBy(F.desc('weight')))) \
                                .filter(f"rn <= %d"%self.max_recommendation_count)  \
                                .groupBy('item_id_i') \
                                .agg(F.collect_list(F.struct(F.col('item_id_j').alias('_1'), F.col('weight').alias('_2'))).alias(self.value_column_name)) \
                                .withColumnRenamed('item_id_i', self.key_column_name)

        if self.debug:
            print('Debug --- item cf result:')
            result.show(10)

        return result

    def _fit(self, dataset):
        dataset = self._filter_dataset(dataset)
        dataset = self._preprocess_dataset(dataset)
        df = self._cf_transform(dataset)
        model = self._create_model(df)
        return model
