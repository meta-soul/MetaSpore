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

import torch
from torch import nn
import metaspore as ms

S3_ROOT_DIR = './'


class WideDeep(torch.nn.Module):
    def __init__(self):
        self.model_id = "WideDeep"
        self.gpu = -1
        self.task = "binary_classification"
        self.learning_rate = 1e-3
        embedding_dim = 10
        self.hidden_unit = [64, 64, 64]
        self.hidden_activations = "ReLU"
        self.net_dropout = 0
        self.batch_norm = False
        self.embedding_regular = None
        self.net_regularizer = None

        super().__init__()
        self.embedding_dim = embedding_dim

        self._schema_dir = S3_ROOT_DIR + 'schema/wdl/'
        self._column_name_path = self._schema_dir+'column_name_demo.txt'
        self._combine_schema_path = self._schema_dir+'combine_schema_demo.txt'

        self._sparse = ms.EmbeddingSumConcat(
            self.embedding_dim, self._column_name_path, self._combine_schema_path)
        self._sparse.updater = ms.FTRLTensorUpdater()
        self._sparse.initializer = ms.NormalTensorInitializer(var=0.01)

        self.final_activation = nn.Sigmoid()

        self.lr_layer = ms.EmbeddingSumConcat(self.embedding_dim,
                                              self._column_name_path,
                                              self._combine_schema_path
                                              )

        self.dnn = torch.nn.Sequential(
            ms.nn.Normalization(
                self._sparse.feature_count * self.embedding_dim),
            torch.nn.Linear(self._sparse.feature_count *
                            self.embedding_dim, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 1),
        )

    def forward(self, x):
        x1 = self.lr_layer(x)
        x1 = x1[:, 0:1]
        x2 = self._sparse(x)
        x2 = self.dnn(x2)
        x1 = torch.add(x1, x2)
        x1 = self.final_activation(x1)
        return x1


module = WideDeep()

model_out_path = S3_ROOT_DIR + 'output/model_out/'
model_export_path = S3_ROOT_DIR + 'output/model_export/'
estimator = ms.PyTorchEstimator(module=module,
                                worker_count=1,
                                server_count=1,
                                experiment_name="wide_and_deep",
                                model_version="1",
                                model_out_path=model_out_path,
                                model_export_path=model_export_path,
                                input_label_column_index=0)

train_dataset_path = S3_ROOT_DIR + 'data/day_0_0.001_train_head1000.csv'
spark_session = ms.spark.get_session(local=True,
                                     batch_size=100,
                                     worker_count=estimator.worker_count,
                                     server_count=estimator.server_count)
train_dataset = ms.input.read_s3_csv(spark_session, train_dataset_path, delimiter='\t')
model = estimator.fit(train_dataset)

test_dataset_path = S3_ROOT_DIR + '/data/day_0_0.001_test_head10.csv'
test_dataset = ms.input.read_s3_csv(spark_session, test_dataset_path, delimiter='\t')
result = model.transform(test_dataset)
result.show()