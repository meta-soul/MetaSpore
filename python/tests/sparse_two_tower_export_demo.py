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

import importlib
import torch
from torch import nn
import torch.nn.functional as F
import metaspore as ms

import sys

sys.path.append('python')

S3_ROOT_DIR = './'


class SimpleXDense(torch.nn.Module):
    def __init__(self, emb_out_size, g):
        super().__init__()
        self._g = g
        self._v = torch.nn.Linear(emb_out_size, emb_out_size, bias=False)
        nn.init.xavier_uniform_(self._v.weight)

    def forward(self, x, y):
        y = self.average_pooling(y)
        z = self._g*x + (1-self._g)*y
        return z

    def average_pooling(self, sequence_emb):
        return self._v(sequence_emb)


class SimilarityModule(torch.nn.Module):
    def __init__(self, net_dropout=0.0):
        super().__init__()
        self.user_dropout = nn.Dropout(
            net_dropout) if net_dropout > 0 else None

    def cosine_similarity_onnx_exportable(x1, x2, dim=-1):
        cross = (x1 * x2).sum(dim=dim)
        x1_l2 = (x1 * x1).sum(dim=dim)
        x2_l2 = (x2 * x2).sum(dim=dim)
        return torch.div(cross, (x1_l2 * x2_l2).sqrt())

    def forward(self, x, y):
        if self.user_dropout is not None:
            x = self.user_dropout(x)
        cosine_similarities = SimilarityModule.cosine_similarity_onnx_exportable(
            x, y, dim=1).reshape(-1, 1)
        return cosine_similarities


class UserModule(torch.nn.Module):
    def __init__(self, column_name_path, user_combine_schema_path, interacted_items_combine_schema_path, emb_size, g, alpha, beta, l1, l2):
        super().__init__()
        self._embedding_size = emb_size
        self._g = g
        self._column_name_path = column_name_path
        self._user_combine_schema_path = user_combine_schema_path
        self._interacted_items_combine_schema_path = interacted_items_combine_schema_path

        self._sparse_user = ms.EmbeddingSumConcat(
            self._embedding_size, self._column_name_path, self._user_combine_schema_path)
        self._sparse_user.updater = ms.FTRLTensorUpdater(
            l1=l1, l2=l2, alpha=alpha, beta=beta)
        self._sparse_user.initializer = ms.NormalTensorInitializer(var=0.0001)
        self._sparse_user.output_batchsize1_if_only_level0 = True

        self._sparse_interacted_items = ms.EmbeddingSumConcat(
            self._embedding_size, self._column_name_path, self._interacted_items_combine_schema_path, embedding_bag_mode='mean')
        self._sparse_interacted_items.updater = ms.FTRLTensorUpdater(
            l1=l1, l2=l2, alpha=alpha, beta=beta)
        self._sparse_interacted_items.initializer = ms.NormalTensorInitializer(
            var=0.0001)
        self._sparse_interacted_items.output_batchsize1_if_only_level0 = True

        self._emb_out_size = self._sparse_user.feature_count * self._embedding_size
        self._dense = SimpleXDense(self._emb_out_size, self._g)

    def forward(self, x):
        a = self._sparse_user(x)
        b = self._sparse_interacted_items(x)
        x = self._dense(a, b)
        x = F.normalize(x)
        return x


class ItemModule(torch.nn.Module):
    def __init__(self, column_name_path, combine_schema_path, emb_size, alpha, beta, l1, l2):
        super().__init__()
        self._embedding_size = emb_size
        self._column_name_path = column_name_path
        self._combine_schema_path = combine_schema_path

        self._sparse = ms.EmbeddingSumConcat(
            self._embedding_size, self._column_name_path, self._combine_schema_path)
        self._sparse.updater = ms.FTRLTensorUpdater(
            l1=l1, l2=l2, alpha=alpha, beta=beta)
        self._sparse.initializer = ms.NormalTensorInitializer(var=0.0001)

        self._emb_out_size = self._sparse.feature_count * self._embedding_size

    def forward(self, x):
        x = self._sparse(x)
        x = F.normalize(x)
        return x


user_column_name = 's3://dmetasoul-bucket/demo/movielens/schema/simplex/user_column_schema'
user_combine_schema = 's3://dmetasoul-bucket/demo/movielens/schema/simplex/user_combine_schema'
interacted_items_combine_schema = 's3://dmetasoul-bucket/demo/movielens/schema/simplex/interacted_items_combine_schema'
item_column_name = 's3://dmetasoul-bucket/demo/movielens/schema/simplex/item_column_schema'
item_combine_schema = 's3://dmetasoul-bucket/demo/movielens/schema/simplex/item_combine_schema'

vector_embedding_size = 32
item_embedding_size = 32
g = 0.8
net_dropout = 0.0
adam_learning_rate = 0.002
ftrl_learning_rate = 0.02
ftrl_smothing_rate = 1.0
ftrl_l1_regularization = 1.0
ftrl_l2_regularization = 1.0

user_module = UserModule(user_column_name, user_combine_schema, interacted_items_combine_schema,
                         emb_size=vector_embedding_size,
                         g=g,
                         alpha=ftrl_learning_rate,
                         beta=ftrl_smothing_rate,
                         l1=ftrl_l1_regularization,
                         l2=ftrl_l2_regularization)
item_module = ItemModule(item_column_name, item_combine_schema,
                         emb_size=vector_embedding_size,
                         alpha=ftrl_learning_rate,
                         beta=ftrl_smothing_rate,
                         l1=ftrl_l1_regularization,
                         l2=ftrl_l2_regularization)
similarity_module = SimilarityModule(net_dropout=0.0)
# import two tower module
module_lib = importlib.import_module("two_tower_retrieval_milvus")
# init module class
module_class_ = getattr(module_lib, "TwoTowerRetrievalModule")
module = module_class_(user_module, item_module, similarity_module)


def prepare_module_save(module):
    # we need to save embedding sizes, names and fe counts
    embedding_size_list = []
    name_list = []
    fe_count_list = []
    for name, mod in module.named_modules():
        if isinstance(mod, ms.embedding.EmbeddingOperator):
            embedding_size_list.append(mod.embedding_size)
            name_list.append(name)
            fe_count_list.append(mod.feature_count)

    return name_list, fe_count_list, embedding_size_list


def extract_dense_module(module, emb_names, emb_fe_count, emb_size):
    module.eval()

    from torch.fx import Tracer, GraphModule

    class MyTracer(Tracer):
        # do not trace through EmbeddingOperator and leave a call_module node
        # otherwise tracer would find sparse outputs as constants and ignore them
        def is_leaf_module(self, m: torch.nn.Module, qualname: str):
            if isinstance(m, ms.embedding.EmbeddingOperator):
                return True
            return super().is_leaf_module(m, qualname)
    my_tracer = MyTracer()

    symbolic_traced: torch.fx.Graph = my_tracer.trace(module)

    new_emb_input_names = []
    new_emb_name_ordered = []
    new_emb_fe_count_ordered = []
    new_emb_emb_size_ordered = []

    for n in symbolic_traced.nodes:
        # remove original placeholders (they are unused in forward)
        if n.op == 'placeholder' and n.name not in new_emb_input_names:
            n._remove_from_list()
        # replace all sparse inputs to dense module as placeholders
        if n.op == 'call_module' and n.target in emb_names:
            index = emb_names.index(n.target)
            new_emb_name_ordered.append(n.target)
            new_emb_fe_count_ordered.append(emb_fe_count[index])
            new_emb_emb_size_ordered.append(emb_size[index])
            new_node = symbolic_traced.placeholder(n.target)
            new_node.target = new_node.name
            new_emb_input_names.append(new_node.name)
            n.replace_all_uses_with(new_node)
            n._remove_from_list()

    symbolic_traced.print_tabular()
    traced_module = GraphModule(
        my_tracer.root, symbolic_traced, 'my_traced_module')
    traced_module.eval()
    return traced_module, new_emb_name_ordered, new_emb_fe_count_ordered, new_emb_emb_size_ordered


def export_module(module, path, model_export_selector):
    if model_export_selector is not None:
        func, _ = model_export_selector
        module = func(module)

    name_list, fe_count_list, embedding_size_list = prepare_module_save(module)
    print(f'Sparse names: {name_list}')

    module, name_list, fe_count_list, embedding_size_list = extract_dense_module(
        module, name_list, fe_count_list, embedding_size_list)
    print(f'Reordered Sparse names: {name_list}')

    script = torch.jit.script(module)

    zero_dim = {0: 'batch_size'}
    dynamic_axes_parameter = {}
    for name in name_list:
        temp = {name: zero_dim}
        dynamic_axes_parameter.update(temp)

    args_parameter = []
    for fe_count, embedding_size in zip(fe_count_list, embedding_size_list):
        args_parameter.append(torch.randn(1, fe_count * embedding_size))

    torch.onnx.export(script, args_parameter,
                      path, input_names=name_list, output_names=[
                          "output"],
                      dynamic_axes=dynamic_axes_parameter,
                      opset_version=14,
                      verbose=True)


# export_module(module, 'two_tower.onnx', None)
export_module(module, 'user_tower_only.onnx',
              (lambda m: m.user_module, '_user_module.'))
