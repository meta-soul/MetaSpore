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

def make_column_name_source(df):
    import pandas as pd
    string = ''
    if isinstance(df, pd.DataFrame):
        for i, column_name in enumerate(df.columns):
            string += '%d %s\n' % (i, column_name)
    else:
        for i, field in enumerate(df.schema.fields):
            string += '%d %s\n' % (i, field.name)
    return string
