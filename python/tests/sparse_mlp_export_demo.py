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
import metaspore as ms

S3_ROOT_DIR = './'

class DenseModule(torch.nn.Module):
    def __init__(self, emb_output_size):
        super().__init__()
        self._dense = torch.nn.Sequential(
            ms.nn.Normalization(emb_output_size),
            torch.nn.Linear(emb_output_size, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 1),
            torch.nn.Sigmoid()
        )
    
    def forward(self, sparse):
        x = self._dense(sparse)
        return x;

class DemoModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._embedding_size = 16
        self._schema_dir = S3_ROOT_DIR + 'schema/'
        self._column_name_path = self._schema_dir + 'column_name_demo.txt'
        self._combine_schema_path = self._schema_dir + 'combine_schema_demo.txt'
        self._sparse0 = ms.EmbeddingSumConcat(
            self._embedding_size, self._column_name_path, self._combine_schema_path)
        self._sparse0.updater = ms.FTRLTensorUpdater()
        self._sparse0.initializer = ms.NormalTensorInitializer(var=0.01)
        self._dense = DenseModule(self._sparse0.feature_count * self._embedding_size)

    def forward(self, x):
        x0 = self._sparse0(x)
        x1 = self._dense(x0)
        return x1


module = DemoModule()
module.eval()

for mod in module.named_children():
    print(mod)

from torch.fx import Tracer, GraphModule 

class MyTracer(Tracer):
    def is_leaf_module(self, m : torch.nn.Module, qualname : str):
        if isinstance(m, ms.embedding.EmbeddingOperator):
            return True
        return super().is_leaf_module(m, qualname)


my_tracer = MyTracer()
# Symbolic tracing frontend - captures the semantics of the module
symbolic_traced : torch.fx.Graph = my_tracer.trace(module)

for n in symbolic_traced.nodes:
    print(n.op, n.name, n.target)
    if n.op == 'placeholder' and 'sparse' not in n.name:
        n._remove_from_list()
    if n.op == 'call_module' and 'sparse' in n.name:
        n._remove_from_list()
        new_node = symbolic_traced.placeholder(n.target)
        new_node.target = new_node.name
        n.replace_all_uses_with(new_node)

# High-level intermediate representation (IR) - Graph representation
# print(symbolic_traced)
symbolic_traced.print_tabular()
traced_module = GraphModule(my_tracer.root, symbolic_traced, 'my_traced_module')
print(traced_module.code)

traced_module.eval()
torch.onnx.export(traced_module, (torch.randn(1, module._sparse0.feature_count * module._embedding_size)),
                  "dense_only_test.onnx", input_names=["_sparse0_1"], output_names=["output"],
                  dynamic_axes={
                      '_sparse0': {0: 'batch_size'},
                  },
                  verbose=True)

# module.train()
# model_export_path = S3_ROOT_DIR + 'model_export/'
# model_out_path = S3_ROOT_DIR + 'model_out/'
# estimator = ms.PyTorchEstimator(module=module,
#                                 worker_count=1,
#                                 server_count=1,
#                                 model_version="1",
#                                 experiment_name="sparse_mlp_exp",
#                                 model_out_path=model_out_path,
#                                 model_export_path=model_export_path,
#                                 input_label_column_index=0)

# spark = ms.spark.get_session(local=True,
#                              batch_size=100,
#                              worker_count=estimator.worker_count,
#                              server_count=estimator.server_count)

# train_dataset_path = S3_ROOT_DIR + '/data/test_sparse_mlp.csv'
# train_dataset = ms.input.read_s3_csv(spark, train_dataset_path, delimiter=',')
# model = estimator.fit(train_dataset)

# test_dataset_path = S3_ROOT_DIR + '/data/test_sparse_mlp.csv'
# test_dataset = ms.input.read_s3_csv(spark, test_dataset_path, delimiter=',')
# result = model.transform(test_dataset)
# result.show()