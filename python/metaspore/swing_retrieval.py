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
from .output import config_cassandra
from .output import ensure_cassandra_db
from .output import write_cassandra

class SwingModel(pyspark.ml.base.Model):
    def __init__(self,
                 df=None,
                 key_column_name='key',
                 value_column_name='value',
                 item_score_delimiter=':',
                 item_score_pair_delimiter=';',
                 item_id_column_name=None,
                 cassandra_catalog=None,
                 cassandra_host_ip=None,
                 cassandra_port=9042,
                 cassandra_user_name=None,
                 cassandra_password=None,
                 cassandra_db_name=None,
                 cassandra_db_properties="class='SimpleStrategy', replication_factor='1'",
                 cassandra_table_name=None,
                ):
        self.df = df
        self.key_column_name = key_column_name
        self.value_column_name = value_column_name
        self.item_score_delimiter = item_score_delimiter
        self.item_score_pair_delimiter = item_score_pair_delimiter
        self.item_id_column_name = item_id_column_name
        self.cassandra_catalog = cassandra_catalog
        self.cassandra_host_ip = cassandra_host_ip
        self.cassandra_port = cassandra_port
        self.cassandra_user_name = cassandra_user_name
        self.cassandra_password = cassandra_password
        self.cassandra_db_name = cassandra_db_name
        self.cassandra_db_properties = cassandra_db_properties
        self.cassandra_table_name = cassandra_table_name

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
        if self.cassandra_catalog is None:
            raise ValueError("cassandra_catalog is required")
        if self.cassandra_host_ip is None:
            raise ValueError("cassandra_host_ip is required")
        if self.cassandra_db_name is None:
            raise ValueError("cassandra_db_name is required")
        if self.cassandra_table_name is None:
            raise ValueError("cassandra_table_name is required")
        spark = self.df.sql_ctx.sparkSession
        config_cassandra(spark, self.cassandra_catalog, self.cassandra_host_ip,
                         port=self.cassandra_port, user_name=self.cassandra_user_name,
                         password=self.cassandra_password)
        ensure_cassandra_db(spark, self.cassandra_catalog, self.cassandra_db_name,
                            db_properties=self.cassandra_db_properties)
        write_cassandra(self.df, self.cassandra_catalog, self.cassandra_db_name,
                        self.cassandra_table_name, partition_key=self.key_column_name)

class SwingEstimator(pyspark.ml.base.Estimator):
    def __init__(self,
                 user_id_column_name=None,
                 item_id_column_name=None,
                 behavior_column_name=None,
                 behavior_filter_value=None,
                 use_plain_weight=False,
                 smoothing_coefficient=1.0,
                 max_recommendation_count=20,
                 key_column_name='key',
                 value_column_name='value',
                 item_score_delimiter=':',
                 item_score_pair_delimiter=';',
                 cassandra_catalog=None,
                 cassandra_host_ip=None,
                 cassandra_port=9042,
                 cassandra_user_name=None,
                 cassandra_password=None,
                 cassandra_db_name=None,
                 cassandra_db_properties="class='SimpleStrategy', replication_factor='1'",
                 cassandra_table_name=None,
                ):
        super().__init__()
        self.user_id_column_name = user_id_column_name
        self.item_id_column_name = item_id_column_name
        self.behavior_column_name = behavior_column_name
        self.behavior_filter_value = behavior_filter_value
        self.use_plain_weight = use_plain_weight
        self.smoothing_coefficient = smoothing_coefficient
        self.max_recommendation_count = max_recommendation_count
        self.key_column_name = key_column_name
        self.value_column_name = value_column_name
        self.item_score_delimiter = item_score_delimiter
        self.item_score_pair_delimiter = item_score_pair_delimiter
        self.cassandra_catalog = cassandra_catalog
        self.cassandra_host_ip = cassandra_host_ip
        self.cassandra_port = cassandra_port
        self.cassandra_user_name = cassandra_user_name
        self.cassandra_password = cassandra_password
        self.cassandra_db_name = cassandra_db_name
        self.cassandra_db_properties = cassandra_db_properties
        self.cassandra_table_name = cassandra_table_name

    def _filter_dataset(self, dataset):
        if self.behavior_column_name is None and self.behavior_filter_value is None:
            return dataset
        if self.behavior_column_name is not None and self.behavior_filter_value is not None:
            return dataset.where(dataset[self.behavior_column_name] == self.behavior_filter_value)
        raise RuntimeError("behavior_column_name and behavior_filter_value must be neither set or both set")

    def _preprocess_dataset(self, dataset):
        import pyspark.sql.functions as F
        if self.user_id_column_name is None:
            raise ValueError("user_id_column_name is required")
        if self.item_id_column_name is None:
            raise ValueError("item_id_column_name is required")
        return (dataset.select(self.user_id_column_name, self.item_id_column_name)
                        .groupBy(dataset[self.user_id_column_name].alias('user'))
                        .agg(F.collect_set(self.item_id_column_name).alias('items')))

    def _get_swing_core_arguments(self):
        return (self.use_plain_weight,
                self.smoothing_coefficient,
                self.max_recommendation_count)

    def _swing_transform(self, dataset):
        from pyspark.sql import DataFrame
        args = self._get_swing_core_arguments()
        jswing = dataset._sc._jvm.com.metaspore.retrievalalgos.SwingCore(*args)
        jdf = jswing.run(dataset._jdf)
        df = DataFrame(jdf, dataset.sql_ctx)
        return df.toDF(self.key_column_name, self.value_column_name)

    def _get_model_arguments(self, df):
        args = dict(df=df,
                    key_column_name=self.key_column_name,
                    value_column_name=self.value_column_name,
                    item_score_delimiter=self.item_score_delimiter,
                    item_score_pair_delimiter=self.item_score_pair_delimiter,
                    item_id_column_name=self.item_id_column_name,
                    cassandra_catalog=self.cassandra_catalog,
                    cassandra_host_ip=self.cassandra_host_ip,
                    cassandra_port=self.cassandra_port,
                    cassandra_user_name=self.cassandra_user_name,
                    cassandra_password=self.cassandra_password,
                    cassandra_db_name=self.cassandra_db_name,
                    cassandra_db_properties=self.cassandra_db_properties,
                    cassandra_table_name=self.cassandra_table_name,
                   )
        return args

    def _create_model(self, df):
        args = self._get_model_arguments(df)
        model = SwingModel(**args)
        return model

    def _fit(self, dataset):
        dataset = self._filter_dataset(dataset)
        dataset = self._preprocess_dataset(dataset)
        df = self._swing_transform(dataset)
        model = self._create_model(df)
        return model
