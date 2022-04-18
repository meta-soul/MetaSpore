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

import xgboost as xgb
import numpy as np
import pathlib
import os

data = np.random.rand(5, 10).astype('f')  # 5 entities, each contains 10 features
label = np.random.randint(2, size=5)  # binary target
dtrain = xgb.DMatrix(data, label=label)

param = {'max_depth': 2, 'eta': 1, 'objective': 'binary:logistic'}
param['nthread'] = 4
param['eval_metric'] = 'auc'

num_round = 10
bst = xgb.train(param, dtrain, num_round, )

from onnxmltools import convert_xgboost
from onnxconverter_common.data_types import FloatTensorType

initial_types = [('input', FloatTensorType(shape=[-1, 10]))]
xgboost_onnx_model = convert_xgboost(bst, initial_types=initial_types, target_opset=14)

output_dir = "output/model_export/xgboost/"

pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

with open(os.path.join(output_dir, 'model.onnx'), "wb") as f:
    f.write(xgboost_onnx_model.SerializeToString())

import onnxruntime as ort
test_data = np.random.rand(1, 10).astype('f')
print(f'Test data: {test_data}')
ort_sess = ort.InferenceSession(os.path.join(output_dir, 'model.onnx'))
outputs = ort_sess.run(None, {'input': test_data})

# Print Result 
print(f'Predicted: "{outputs}"')