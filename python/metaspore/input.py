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

def shuffle_df(df, num_workers, column_name='srand'):
    from pyspark.sql import functions as F
    df = df.withColumn(column_name, F.rand())
    df = df.repartition(2 * num_workers, column_name)
    print('shuffle df to partitions {}'.format(df.rdd.getNumPartitions()))
    df = df.sortWithinPartitions(column_name)
    df = df.drop(column_name)
    return df

def read_s3_csv(spark_session, url, shuffle=False, num_workers=1,
                header=False, nullable=False, delimiter="\002", multivalue_delimiter="\001",
                encoding="UTF-8", schema=None, column_names=None, multivalue_column_names=None):
    from .url_utils import use_s3a
    from .schema_utils import make_csv_schema
    from .schema_utils import make_csv_transformer
    schema = make_csv_schema(schema, column_names, multivalue_column_names)
    input_schema, df_transformer = make_csv_transformer(schema, multivalue_delimiter)
    df = (spark_session
             .read
             .format('csv')
             .option("header", str(bool(header)).lower())
             .option("nullable", str(bool(nullable)).lower())
             .option("delimiter", delimiter)
             .option("encoding", encoding)
             .schema(input_schema)
             .load(use_s3a(url)))
    df = df_transformer(df)
    if shuffle and num_workers > 1:
        df = shuffle_df(df, num_workers)
    else:
        print("ignore shuffle")
    return df

def read_s3_image(spark_session, url):
    from .url_utils import use_s3a
    df = spark_session.read.format('image').option('dropInvalid', 'true').load(use_s3a(url))
    return df
