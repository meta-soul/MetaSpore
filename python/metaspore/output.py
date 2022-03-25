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

def write_s3_csv(df, url, mode="overwrite",
                 header=False, delimiter="\002", encoding="UTF-8"):
    from .url_utils import use_s3a
    df.write.csv(use_s3a(url), mode=mode, header=header, sep=delimiter, encoding=encoding)

def config_cassandra(spark_session, catalog, host_ip, port=9042, user_name=None, password=None):
    catalog_key = f'spark.sql.catalog.{catalog}'
    catalog_value = 'com.datastax.spark.connector.datasource.CassandraCatalog'
    host_key = f'{catalog_key}.spark.cassandra.connection.host'
    port_key = f'{catalog_key}.spark.cassandra.connection.port'
    user_name_key = f'{catalog_key}.spark.cassandra.auth.username'
    password_key = f'{catalog_key}.spark.cassandra.auth.password'
    spark_session.conf.set(catalog_key, catalog_value)
    spark_session.conf.set(host_key, host_ip)
    spark_session.conf.set(port_key, str(port))
    if user_name is not None:
        spark_session.conf.set(user_name_key, user_name)
    if password is not None:
        spark_session.conf.set(password_key, password)

def ensure_cassandra_db(spark_session, catalog, db_name,
                        db_properties="class='SimpleStrategy', replication_factor='1'"):
    spark_session.sql(f'CREATE DATABASE IF NOT EXISTS {catalog}.{db_name} '
                      f'WITH DBPROPERTIES ({db_properties})')

def write_cassandra(df, catalog, db_name, table_name, partition_key='key', mode='overwrite'):
    table = f'{catalog}.{db_name}.{table_name}'
    df.write.partitionBy(partition_key).mode(mode).saveAsTable(table)
