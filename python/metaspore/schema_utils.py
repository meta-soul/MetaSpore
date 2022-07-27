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

def make_csv_schema(schema, column_names, multivalue_column_names):
    from pyspark.sql.types import StructType
    from pyspark.sql.types import StructField
    from pyspark.sql.types import StringType
    from pyspark.sql.types import ArrayType
    if schema is None and column_names is None:
        message = "one of schema and column_names must be specified"
        raise RuntimeError(message)
    if schema is not None and column_names is not None:
        message = "both schema and column_names are specified"
        raise RuntimeError(message)
    if schema is None:
        if multivalue_column_names is None:
            multivalue_column_names = frozenset()
        else:
            multivalue_column_names = frozenset(multivalue_column_names)
        fields = []
        for name in column_names:
            if name in multivalue_column_names:
                fields.append(StructField(name, ArrayType(StringType())))
            else:
                fields.append(StructField(name, StringType()))
        schema = StructType(fields)
    return schema

def is_data_type_supported(data_type):
    from pyspark.sql.types import StringType
    from pyspark.sql.types import FloatType
    from pyspark.sql.types import DoubleType
    from pyspark.sql.types import IntegerType
    from pyspark.sql.types import LongType
    from pyspark.sql.types import BooleanType
    from pyspark.sql.types import ArrayType
    types = (StringType, FloatType, DoubleType, IntegerType, LongType, BooleanType)
    if isinstance(data_type, types):
        return True
    if isinstance(data_type, ArrayType) and isinstance(data_type.elementType, types):
        return True
    return False

def make_csv_transformer(schema, multivalue_delimiter):
    import pyspark.sql.functions as F
    from pyspark.sql.types import StructType
    from pyspark.sql.types import StructField
    from pyspark.sql.types import StringType
    from pyspark.sql.types import ArrayType
    fields = []
    expressions = []
    for field in schema:
        if not is_data_type_supported(field.dataType):
            message = "data type of column %r is not supported" % field.name
            raise RuntimeError(message)
        if isinstance(field.dataType, ArrayType):
            fields.append(StructField(field.name, StringType(), field.nullable))
            expressions.append(F.split(F.col(field.name), multivalue_delimiter)
                                .cast(field.dataType)
                                .alias(field.name))
        else:
            fields.append(field)
            expressions.append(F.col(field.name))
    input_schema = StructType(fields)
    df_transformer = lambda df: df.select(*expressions)
    return input_schema, df_transformer
