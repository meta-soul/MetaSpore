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

import numpy as np

from pyspark.sql.functions import col
from pyspark.sql.types import StructType, StructField, StringType

def get_crieto_meta():
    ''' Generate meta data for the datast according to description in: https://www.csie.ntu.edu.tw/~r01922136/kaggle-2014-criteo.pdf
    '''

    headers = ["label", "I1", "I2", "I3", "I4", "I5", "I6", "I7", "I8", "I9", "I10",
           "I11", "I12", "I13", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10",
           "C11", "C12", "C13", "C14", "C15", "C16", "C17", "C18", "C19", "C20", "C21", "C22", 
           "C23", "C24", "C25", "C26"]
    schema = StructType([StructField(h, StringType(), True) for h in headers])
    return headers, schema

def read_crieto_files(s3_root_dir, days=1, mode='train'):
    ''' Read criteo files from S3 file systems

    Args
      - s3_root_dir: root directory of the original dataset
      - days: how many days will including to generate training and test datasets
      - mode: train or test
    '''

    paths = [s3_root_dir + '%s/day_%d_0.001_%s.csv'%(mode, x, mode) for x in range(0, days)]
    print('Debug read_crieto_files -- paths: %s' % paths)
    headers, schema = get_crieto_meta()
    dataset = spark.read.csv(paths, sep='\t',inferSchema=False, header=False, schema=schema)
    print('Debug read_crieto_files -- data count: %d' % dataset.count())
    return dataset

def transform_number(x):
    ''' For numeric features of I1~I13, greater than 2 are transformed as below

    Args
      x: original numeric feature value
    '''

    value = -1
    try:
        if x is not None:
            value = float(x)
    except ValueError:
        pass
    return int(np.floor(np.log(value) ** 2)) if value>2.0 else int(value)

def transform(row):
    ''' Transform one row of the pyspark dataframe

    Args
      - row: one row of pyspark dataframe
    '''

    row_list = []
    for k, v in row.asDict().items():
        if k.startswith('I') :
            row_list.append(transform_number(v))
        elif k.startswith('C'):
            row_list.append(str(v))
        else:
            row_list.append(v)
    return row_list

def feature_generation(dataset, debug=False):
    ''' PySpark version of feature generation of `3 Idiots' Approach` described in https://www.csie.ntu.edu.tw/~r01922136/kaggle-2014-criteo.pdf

    Args
      - dataset: original dataset provided in Display Advertising Challenge
    '''

    dataset = dataset.rdd.map(lambda x: transform(x)).toDF(dataset.schema.names)
    dataset = dataset.select(*(col(c).cast('string').alias(c) for c in dataset.columns))  
    return dataset    
