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

import functools

from pyspark.sql import functions as F

def gen_numerical_features(dataset, label_col, cate_cols_list, combine_sep='#'):
    features = []
    features_list = []
    for cols in cate_cols_list:
        for col in cols:
            col_list = col.split(combine_sep)
            df_cr_pv = dataset.groupBy(*col_list)\
                .agg((F.sum(label_col) / F.count(label_col)).alias('_'.join(col_list) + '_cr'),
                      F.sum(label_col).alias('_'.join(col_list) + '_pv'))
            dataset = dataset.join(df_cr_pv, on=col_list, how='left_outer')
            df_cr_pv = df_cr_pv.select(df_cr_pv.colRegex("`^user_id$|^item_id$|^.*_(cr|pv)$`"))
            features = features + [f.name for f in df_cr_pv.schema.fields]
        features_list.append(features)
        features = []
    dataset = dataset.select(
        ['label'] + \
        functools.reduce(lambda x, y: x+y, features_list)
    )
    return dataset, features_list
