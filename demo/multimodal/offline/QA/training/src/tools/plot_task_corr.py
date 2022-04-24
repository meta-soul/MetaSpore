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

import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

eval_file = sys.argv[1]
df = pd.read_csv(eval_file, sep='\t', index_col=0)
#converters = {c:lambda x: float(x.strip('%')) for c in df.columns.tolist()}
other_rows = list(set(df.index.tolist())-set(df.columns.tolist()))
df1 = df.loc[df.columns.tolist()]
df = df1.append(df.loc[other_rows])

plt.figure(figsize = (15,8))
ax = sns.heatmap(df, vmin=0.0, vmax=1.0, annot=True, fmt='.1%', cmap='YlOrRd')
#ax.set_xticklabels(ax.get_xticklabels(),rotation = 30)
#plt.xticks(rotation=20)

#plt.savefig('output.png')
fig = ax.get_figure()
fig.savefig('output.png', dpi=300)
