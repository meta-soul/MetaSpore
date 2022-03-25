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
        )
        self._act = torch.nn.Sigmoid()
    
    def forward(self, wide, deep):
        deep = self._dense(deep)
        sum = torch.add(wide, deep)
        return self._act(sum)

class DemoModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._embedding_size = 16
        self._schema_dir = S3_ROOT_DIR + 'schema/wdl/'
        self._column_name_path = self._schema_dir + 'column_name_demo.txt'
        self._combine_schema_path = self._schema_dir + 'combine_schema_demo.txt'
        self.lr_layer = ms.EmbeddingSumConcat(
            self._embedding_size, self._column_name_path, self._combine_schema_path)
        self.deep_layer = ms.EmbeddingSumConcat(
            self._embedding_size, self._column_name_path, self._combine_schema_path)
        self.lr_layer.updater = ms.FTRLTensorUpdater()
        self.lr_layer.initializer = ms.NormalTensorInitializer(var=0.01)
        self.deep_layer.updater = ms.FTRLTensorUpdater()
        self.deep_layer.initializer = ms.NormalTensorInitializer(var=0.01)
        self._dense = DenseModule(self.lr_layer.feature_count * self._embedding_size)

    def forward(self, x):
        x0 = self.lr_layer(x)
        x1 = self.deep_layer(x)
        x3 = self._dense(x0, x1)
        return x3


module = DemoModule()
module.eval()

emb_names = []

for name, mod in module.named_children():
    if isinstance(mod, ms.embedding.EmbeddingOperator):
        emb_names.append(name)

from torch.fx import Tracer, GraphModule 

class MyTracer(Tracer):
    def is_leaf_module(self, m : torch.nn.Module, qualname : str):
        if isinstance(m, ms.embedding.EmbeddingOperator):
            return True
        return super().is_leaf_module(m, qualname)


my_tracer = MyTracer()
# Symbolic tracing frontend - captures the semantics of the module
symbolic_traced : torch.fx.Graph = my_tracer.trace(module)

new_input_names = []

for n in symbolic_traced.nodes:
    print(n.op, n.name, n.target)
    if n.op == 'placeholder' and n.name not in new_input_names:
        n._remove_from_list()
    if n.op == 'call_module' and n.name in emb_names:
        new_node = symbolic_traced.placeholder(n.target)
        new_node.target = new_node.name
        new_input_names.append(new_node.name)
        n.replace_all_uses_with(new_node)
        n._remove_from_list()

# High-level intermediate representation (IR) - Graph representation
# print(symbolic_traced)
symbolic_traced.print_tabular()
traced_module = GraphModule(my_tracer.root, symbolic_traced, 'my_traced_module')
print(traced_module.code)

traced_module.eval()
script = torch.jit.script(traced_module)
torch.onnx.export(script, (torch.randn(1, module.lr_layer.feature_count * module._embedding_size),
                           torch.randn(1, module.deep_layer.feature_count * module._embedding_size)),
                  "dense_only_test.onnx", input_names=new_input_names, output_names=["output"],
                  dynamic_axes={
                      new_input_names[0]: {0: 'batch_size'},
                      new_input_names[1]: {0: 'batch_size'}
                  },
                  verbose=True)