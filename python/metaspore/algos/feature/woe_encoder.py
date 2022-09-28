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

import pyspark.sql.functions as F

class WoeEncoder(object):
    def __init__(self, colunms_to_woe, label_col, bad_label_value=1.0, output_suffix='_woe'):
        self.colunms_to_woe = colunms_to_woe
        self.label_col = label_col
        self.bad_label_value = bad_label_value
        self.output_suffix = output_suffix
        self.data = {}

    def fit(self, dataframe):
        total_bad = dataframe.filter(F.col(self.label_col) == self.bad_label_value).count()
        total_good = dataframe.filter(F.col(self.label_col) != self.bad_label_value).count()
        for colunm_to_woe in self.colunms_to_woe:
            df = dataframe.groupBy(F.col(colunm_to_woe)).agg(F.count(colunm_to_woe).alias('total_in_bucket'), F.sum(self.label_col).alias('label_sum'))
            if self.bad_label_value == 1:
                df = df.withColumnRenamed('label_sum', 'bad_in_bucket')
                df = df.withColumn('good_in_bucket', df.total_in_bucket - df.bad_in_bucket)
            elif self.bad_label_value == 0:
                df = df.withColumnRenamed('label_sum', 'good_in_bucket')
                df = df.withColumn('bad_in_bucket', df.total_in_bucket - df.good_in_bucket)
            else:
                raise Exception('bad_label_value must be 1.0 or 0.0, but current value is: ', bad_label_value)
            df = df.withColumn('bad_in_bucket_fixed', F.when(df.bad_in_bucket > 0, df.bad_in_bucket).otherwise(0.5))
            df = df.withColumn('good_in_bucket_fixed', F.when(df.good_in_bucket > 0, df.good_in_bucket).otherwise(0.5))
            df = df.withColumn('woe', F.log((df.bad_in_bucket_fixed / total_bad) / (df.good_in_bucket_fixed / total_good)))

            col_dict = {row[colunm_to_woe]:row['woe'] for row in df.collect()}
            self.data[colunm_to_woe] = col_dict

        return self.data

    def transform(self, dataframe):
        for colunm_to_woe in self.colunms_to_woe:
            dataframe = dataframe.withColumn(colunm_to_woe + self.output_suffix, F.coalesce(*[F.when(F.col(colunm_to_woe) == key, F.lit(value)) for key, value in self.data[colunm_to_woe].items()]))

        return dataframe
