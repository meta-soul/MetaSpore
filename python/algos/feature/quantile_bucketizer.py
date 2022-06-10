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
from pyspark.ml.feature import QuantileDiscretizer

def quantile_bucket(dataframe, input_cols, num_buckets, output_suffix='_bucket', keep_both=False):
    output_cols = [ic + output_suffix for ic in input_cols]
    discretizer = QuantileDiscretizer(numBuckets=num_buckets, inputCols=input_cols, outputCols=output_cols, handleInvalid='skip')
    df = discretizer.fit(dataframe).transform(dataframe)
    if not keep_both:
        df = df.drop(*input_cols)
        df = df.select(*(F.col(c).alias(c.replace(output_suffix, '')) if c in output_cols else F.col(c) for c in df.columns))
    return df