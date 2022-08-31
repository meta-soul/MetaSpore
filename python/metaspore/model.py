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

import os
import json
import time
import asyncio
import torch
import collections
from . import _metaspore
from . import embedding
from .agent import Agent
from .name_utils import is_valid_qualified_name
from .initializer import ZeroTensorInitializer
from .initializer import OneTensorInitializer
from .updater import EMATensorUpdater
from .embedding import EmbeddingOperator
from .cast import Cast
from .distributed_tensor import DistributedTensor
from .url_utils import use_s3


class Model(object):
    def __init__(self, agent, module, experiment_name=None, model_version=None, name_prefix=None):
        if not isinstance(agent, Agent):
            raise TypeError(f"agent must be Agent; {agent!r} is invalid")
        if not isinstance(module, torch.nn.Module):
            raise TypeError(
                f"module must be torch.nn.Module; {module!r} is invalid")
        if experiment_name is not None:
            if not isinstance(experiment_name, str) or not is_valid_qualified_name(experiment_name.strip()):
                raise TypeError(
                    f"experiment_name must be valid qualified name; {experiment_name!r} is invalid")
            experiment_name = experiment_name.strip()
        if model_version is not None:
            if not isinstance(model_version, str):
                raise TypeError(
                    f"model_version must be string; {model_version!r} is invalid")
            model_version = model_version.strip()
        if name_prefix is not None:
            if not isinstance(name_prefix, str):
                raise TypeError(
                    f"name_prefix must be string; {name_prefix!r} is invalid")
        self._agent = agent
        self._module = module
        self._experiment_name = experiment_name
        self._model_version = model_version
        self._name_prefix = name_prefix
        self._tensors = []

    @property
    def agent(self):
        return self._agent

    @property
    def module(self):
        return self._module

    @property
    def experiment_name(self):
        return self._experiment_name

    @experiment_name.setter
    def experiment_name(self, value):
        if value is not None:
            if not isinstance(value, str) or not is_valid_qualified_name(value.strip()):
                raise TypeError(
                    f"experiment_name must be valid qualified name; {value!r} is invalid")
            value = value.strip()
        if self._experiment_name is not None:
            raise RuntimeError(
                f"can not reset experiment_name {self._experiment_name!r} to {value!r}")
        self._experiment_name = value

    def _checked_get_experiment_name(self):
        if self._experiment_name is None:
            raise RuntimeError("experiment_name is not set")
        return self._experiment_name

    @property
    def model_version(self):
        return self._model_version

    @model_version.setter
    def model_version(self, value):
        if value is not None:
            if not isinstance(value, str):
                raise TypeError(
                    f"model_version must be string; {value!r} is invalid")
            value = value.strip()
        if self._model_version is not None:
            raise RuntimeError(
                f"can not reset model_version {self._model_version!r} to {value!r}")
        self._model_version = value

    @property
    def name_prefix(self):
        return self._name_prefix

    @name_prefix.setter
    def name_prefix(self, value):
        if value is not None:
            if not isinstance(value, str):
                raise TypeError(
                    f"name_prefix must be string; {value!r} is invalid")
        if self._name_prefix is not None:
            raise RuntimeError(
                f"can not reset name_prefix {self._name_prefix!r} to {value!r}")
        self._name_prefix = value

    @property
    def training(self):
        return self._module.training

    def train(self):
        if not self.training:
            self._module.train()

    def eval(self):
        if self.training:
            self._module.eval()

    @classmethod
    def _filter_tensor_list(cls, tensors, name_prefix):
        return [t for t in tensors if t.name.startswith(name_prefix)]

    def get_submodel(self, submodule, name_prefix):
        submodel = self.__new__(self.__class__)
        submodel._agent = self._agent
        submodel._module = submodule
        submodel._experiment_name = self._experiment_name
        submodel._model_version = self._model_version
        submodel._name_prefix = self._name_prefix
        submodel._tensors = self._filter_tensor_list(
            self._tensors, name_prefix)
        return submodel

    def _is_batch_norm(self, name, mod):
        if isinstance(mod, torch.nn.modules.batchnorm._BatchNorm):
            if isinstance(mod, torch.nn.SyncBatchNorm):
                message = f"module {name!r} is an instance of torch.nn.SyncBatchNorm, "
                message += "which is not supported by DistributedTrainer"
                raise RuntimeError(message)
            return True
        return False

    def _configure_batch_norm(self, name, mod):
        if mod.running_mean is not None and getattr(mod.running_mean, 'initializer', None) is None:
            mod.running_mean.initializer = ZeroTensorInitializer()
        if mod.running_mean is not None and getattr(mod.running_mean, 'updater', None) is None:
            mod.running_mean.updater = EMATensorUpdater()
        if mod.running_var is not None and getattr(mod.running_var, 'initializer', None) is None:
            mod.running_var.initializer = OneTensorInitializer()
        if mod.running_var is not None and getattr(mod.running_var, 'updater', None) is None:
            mod.running_var.updater = EMATensorUpdater()
        if mod.weight is not None and getattr(mod.weight, 'initializer', None) is None:
            mod.weight.initializer = OneTensorInitializer()
        if mod.bias is not None and getattr(mod.bias, 'initializer', None) is None:
            mod.bias.initializer = ZeroTensorInitializer()

    def _configure_batch_norms(self):
        for name, mod in self.module.named_modules():
            if self._is_batch_norm(name, mod):
                self._configure_batch_norm(name, mod)

    def _collect_batch_norm(self, name, mod):
        running_mean = mod.running_mean
        running_mean_name = name + '.running_mean'
        running_var = mod.running_var
        running_var_name = name + '.running_var'
        if running_mean is None and running_var is None:
            return
        if running_mean is not None and running_var is not None:
            tensor1 = DistributedTensor(
                running_mean_name, running_mean, self.name_prefix)
            tensor2 = DistributedTensor(
                running_var_name, running_var, self.name_prefix)
            self._tensors.append(tensor1)
            self._tensors.append(tensor2)
            return
        message = "running_mean and running_var must be both None or both not None; "
        message += f"BatchNorm module {name!r} is invalid"
        raise RuntimeError(message)

    def _collect_embedding_operators(self):
        for name, mod in self.module.named_modules():
            if isinstance(mod, EmbeddingOperator):
                message = f"embedding operator {name!r} detected, "
                message += "please use SparseModel instead of Model as the wrapper"
                raise RuntimeError(message)

    def _collect_cast_operators(self):
        for name, mod in self.module.named_modules():
            if isinstance(mod, Cast):
                message = f"cast operator {name!r} detected, "
                message += "please use SparseModel instead of Model as the wrapper"
                raise RuntimeError(message)

    def _collect_dense_parameters(self):
        for name, param in self.module.named_parameters():
            tensor = DistributedTensor(name, param, self.name_prefix)
            self._tensors.append(tensor)

    def _collect_dense_buffers(self):
        for name, mod in self.module.named_modules():
            if self._is_batch_norm(name, mod):
                self._collect_batch_norm(name, mod)

    def _collect_tensors(self):
        self._collect_embedding_operators()
        self._collect_cast_operators()
        self._collect_dense_parameters()
        self._collect_dense_buffers()

    async def _init_tensors(self, trainer):
        futures = []
        for tensor in self._tensors:
            future = tensor._init_tensor(trainer)
            futures.append(future)
        await asyncio.gather(*futures)

    async def _pull_tensors(self, *, force_mode=False):
        futures = []
        for tensor in self._tensors:
            if not force_mode:
                # Pulling dense parameters in prediction mode is redundant.
                if not self.training and tensor.is_dense:
                    continue
            if not tensor.is_backing:
                future = tensor._pull_tensor()
                futures.append(future)
        await asyncio.gather(*futures)

    async def _push_tensors(self, *, is_value=False, skip_no_grad=True):
        futures = []
        for tensor in self._tensors:
            if not tensor.is_backing:
                future = tensor._push_tensor(
                    is_value=is_value, skip_no_grad=skip_no_grad)
                futures.append(future)
        await asyncio.gather(*futures)

    async def _clear_tensors(self):
        pass

    async def _load_tensors(self, dir_path, *, keep_meta=False):
        futures = []
        for tensor in self._tensors:
            if not tensor.is_backing:
                future = tensor._load_tensor(dir_path, keep_meta=keep_meta)
                futures.append(future)
        await asyncio.gather(*futures)

    async def _save_tensors(self, dir_path):
        futures = []
        for tensor in self._tensors:
            if not tensor.is_backing:
                future = tensor._save_tensor(dir_path)
                futures.append(future)
        await asyncio.gather(*futures)

    def prune_small(self, epsilon=1e-6):
        if not isinstance(epsilon, float) or epsilon < 0.0:
            if epsilon != 0:
                raise TypeError(
                    f"epsilon must be non-negative float or 0; {epsilon!r} is invalid")
        self._do_prune_small(epsilon)

    def prune_old(self, max_age):
        if not isinstance(max_age, int) or max_age <= 0:
            raise TypeError(
                f"max_age must be positive integer; {max_age!r} is invalid")
        self._do_prune_old(max_age)

    def _do_prune_small(self, epsilon):
        pass

    def _do_prune_old(self, max_age):
        pass

    def _get_full_class_name(self, obj):
        cls = obj.__class__
        name = '%s.%s' % (cls.__module__, cls.__name__)
        return name

    def _get_model_version(self):
        if self._model_version is not None:
            return self._model_version
        # default to a string in Beijing Time (UTC+8)
        now = time.time()
        now += 8 * 3600
        tm = time.gmtime(now)
        ver = time.strftime('%Y%m%d%H', tm)
        return ver

    def _get_export_meta(self, path, *, model_export_selector=None):
        meta_version = 1
        agent_class = self._get_full_class_name(self.agent)
        model_class = self._get_full_class_name(self)
        model_version = self._get_model_version()
        module = self.module
        if model_export_selector is not None:
            func, name_prefix_ = model_export_selector
            module = func(module)
        module_class = self._get_full_class_name(module)
        module_file = os.path.basename(path)
        experiment_name = self._checked_get_experiment_name()
        meta = {
            'meta_version': meta_version,
            'agent_class': agent_class,
            'model_class': model_class,
            'model_version': model_version,
            'module_class': module_class,
            'module_file': module_file,
            'experiment_name': experiment_name,
        }
        return meta

    def _from_json_string(self, string):
        return json.loads(string, object_pairs_hook=collections.OrderedDict)

    def _as_json_string(self, obj):
        string = json.dumps(obj, separators=(',', ': '), indent=4)
        return string

    def forward(self):
        return

    # extract model output names or use user specified names
    def _extract_model_output_names(self, graph, *, output_names=None):
        output_nodes = [n for n in graph.nodes if n.op == 'output']
        if len(output_nodes) != 1:
            message = f"exactly one output node expected, found {len(output_nodes)}"
            raise RuntimeError(message)
        node = output_nodes[0]
        args = node.args[0]
        if not isinstance(args, tuple):
            args = args,
        names = [n.name for n in args]
        if output_names is None:
            if len(names) == 1:
                return ['output']
            else:
                return ['output_%d' % (i + 1) for i in range(len(names))]
        else:
            output_names = list(output_names)
            if len(names) != len(output_names):
                message = f"user specified {len(output_names)} output names {output_names}, "
                message += f"but found {len(names)} {names}"
                raise RuntimeError(message)
            return output_names

    # extract when the module has multiple input eg: wide and deep
    def _extract_dense_module(self, module, emb_names, emb_fe_count, emb_size, *, output_names=None):
        module.eval()

        from torch.fx import Tracer, GraphModule

        class MyTracer(Tracer):
            # do not trace through EmbeddingOperator and leave a call_module node
            # otherwise tracer would find sparse outputs as constants and ignore them
            def is_leaf_module(self, m: torch.nn.Module, qualname: str):
                if isinstance(m, embedding.EmbeddingOperator):
                    return True
                return super().is_leaf_module(m, qualname)
        my_tracer = MyTracer()

        symbolic_traced: torch.fx.Graph = my_tracer.trace(module)

        new_emb_input_names = []
        new_emb_name_ordered = []
        new_emb_fe_count_ordered = []
        new_emb_emb_size_ordered = []

        for n in symbolic_traced.nodes:
            print(f'node {n.op}, {n.name}, {n.target}')
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
        output_names = self._extract_model_output_names(symbolic_traced, output_names=output_names)
        traced_module = GraphModule(
            my_tracer.root, symbolic_traced, 'my_traced_module')
        traced_module.eval()
        return traced_module, new_emb_name_ordered, new_emb_fe_count_ordered, new_emb_emb_size_ordered, output_names

    def _prepare_module_save(self, model_export_selector=None):
        module = self.module
        name_prefix = None
        if model_export_selector is not None:
            func, name_prefix = model_export_selector
            module = func(module)
        selected = set(module.modules())

        # we need to save embedding sizes, names and fe counts
        embedding_size_list = []
        name_list = []
        fe_count_list = []
        for tensor in self._embedding_operators:
            # Call the ``_clean()`` method so that intermediate results
            # in the EmbeddingOperator won't be serialized.
            tensor.item._clean()
            if tensor.item in selected and tensor.is_exported:
                name = tensor.name
                if name_prefix is not None:
                    if not name.startswith(name_prefix):
                        message = f"tensor name {name!r} mismatches with name prefix {name_prefix!r}"
                        raise RuntimeError(message)
                    name = name[len(name_prefix):]
                embedding_size_list.append(tensor.item.embedding_size)
                name_list.append(name)
                fe_count_list.append(tensor.item.feature_count)

        return name_list, fe_count_list, embedding_size_list

    def _do_export(self, path, *, model_export_selector=None, output_names=None):
        # change the dense dir
        path = os.path.join(use_s3(path), '_dense/model.onnx')

        module = self.module
        if model_export_selector is not None:
            func, name_prefix_ = model_export_selector
            module = func(module)

        name_list, fe_count_list, embedding_size_list = self._prepare_module_save(
            model_export_selector)
        print(f'name_list {name_list}')

        # use the torch fx Tracer, GraphModule
        module, name_list, fe_count_list, embedding_size_list, output_names = self._extract_dense_module(
            module, name_list, fe_count_list, embedding_size_list, output_names=output_names)

        script = torch.jit.script(module)
        dir_path = os.path.dirname(path)
        _metaspore.ensure_local_directory(dir_path)

        # use a flush to fake the onnx output
        class FakeStream(object):
            def write(self, data):
                _metaspore.stream_write_all(path, data)

            def flush(self):
                pass
        fout = FakeStream()

        # Multiple input, eg: the wide and deep module
        zero_dim = {0: 'batch_size'}
        dynamic_axes_parameter = {}
        for name in name_list:
            temp = {name: zero_dim}
            dynamic_axes_parameter.update(temp)

        args_parameter = []
        for fe_count, embedding_size in zip(fe_count_list, embedding_size_list):
            args_parameter.append(torch.randn(1, fe_count * embedding_size))

        torch.onnx.export(script, args_parameter,
                          fout, input_names=name_list, output_names=output_names,
                          dynamic_axes=dynamic_axes_parameter,
                          opset_version=14,
                          verbose=True)

    def export(self, path, *, model_export_selector=None, output_names=None):
        if not isinstance(path, str) or not path.strip():
            raise TypeError(f"path must be non-empty string; {path!r} is invalid")
        path = path.strip()
        if not path.endswith('/'):
            raise ValueError(f"path must be directory path endswith /; {path!r} is invalid")
        if self._experiment_name is None:
            raise RuntimeError(f"experiment_name is not set; can not export to {path!r}")
        if self.training:
            message = "model is in training mode, can not export it; "
            message += "call the 'eval' method to set it in evaluation mode explicitly"
            raise RuntimeError(message)
        self.agent.barrier()
        asyncio.run(self._pull_tensors(force_mode=True))
        if self.agent.rank == 0:
            self._do_export(path, model_export_selector=model_export_selector, output_names=output_names)
        self.agent.barrier()

    def sync(self):
        self.agent.barrier()
        asyncio.run(self._pull_tensors(force_mode=True))
        self.agent.barrier()

    def __call__(self, *inputs):
        # Pulling dense parameters in prediction mode is redundant.
        if self.training:
            asyncio.run(self._pull_tensors())
        return self.module(*inputs)

    def _zero_grad(self):
        for tensor in self._tensors:
            tensor._zero_grad()

    @classmethod
    def _contains_embedding_operators(cls, module):
        for name, mod in module.named_modules():
            if isinstance(mod, EmbeddingOperator):
                return True
        return False

    @classmethod
    def _contains_cast_operators(cls, module):
        for name, mod in module.named_modules():
            if isinstance(mod, Cast):
                return True
        return False

    @classmethod
    def wrap(cls, agent, module, experiment_name=None, model_version=None, name_prefix=None):
        if (cls._contains_embedding_operators(module) or
                cls._contains_cast_operators(module)):
            return SparseModel(agent, module, experiment_name, model_version, name_prefix)
        else:
            return Model(agent, module, experiment_name, model_version, name_prefix)


class SparseModel(Model):
    def __init__(self, agent, module, experiment_name=None, model_version=None, name_prefix=None):
        super().__init__(agent, module, experiment_name, model_version, name_prefix)
        self._embedding_operators = []
        self._cast_operators = []

    def get_submodel(self, submodule, name_prefix):
        submodel = super().get_submodel(submodule, name_prefix)
        submodel._embedding_operators = self._filter_tensor_list(
            self._embedding_operators, name_prefix)
        submodel._cast_operators = self._filter_tensor_list(
            self._cast_operators, name_prefix)
        return submodel

    def _collect_embedding_operators(self):
        for name, mod in self.module.named_modules():
            if isinstance(mod, EmbeddingOperator):
                tensor = DistributedTensor(name, mod, self.name_prefix)
                self._tensors.append(tensor)
                self._embedding_operators.append(tensor)
                mod._distributed_tensor = tensor

    def _collect_cast_operators(self):
        for name, mod in self.module.named_modules():
            if isinstance(mod, Cast):
                self._cast_operators.append(mod)

    async def _clear_tensors(self):
        futures = []
        for tensor in self._embedding_operators:
            if not tensor.is_backing:
                future = tensor._sparse_tensor_clear()
                futures.append(future)
        await asyncio.gather(*futures)

    async def _sparse_tensors_export(self, path, *, model_export_selector=None):
        futures = []
        module = self.module
        name_prefix = None
        if model_export_selector is not None:
            func, name_prefix = model_export_selector
            module = func(module)
        selected = set(module.modules())

        for tensor in self._embedding_operators:
            # Call the ``_clean()`` method so that intermediate results
            # in the EmbeddingOperator won't be serialized.
            tensor.item._clean()
            if tensor.item in selected and tensor.is_exported:
                name = tensor.name
                if name_prefix is not None:
                    if not name.startswith(name_prefix):
                        message = f"tensor name {name!r} mismatches with name prefix {name_prefix!r}"
                        raise RuntimeError(message)
                    name = name[len(name_prefix):]

                dir = 'sparse_' + name
                # save sparse onnx, which confer to the embedding.py: see the method in embedding.py
                sparse_onnx_dir = path + '/' + dir + '/' + 'model.onnx'
                tensor.item.export_sparse_embedding_bag(sparse_onnx_dir, name)

                # copy the schema
                data = tensor.item.combine_schema_source.encode('utf-8')
                _metaspore.stream_write_all(
                    use_s3(path + '/' + dir + '/' + 'combine_schema.txt'), data)

                # save embedding table, we just need to change the dir
                dir_path = path + '/' + dir + '/' + 'embedding_table/'

                future = tensor._sparse_tensor_export(dir_path)
                futures.append(future)
        await asyncio.gather(*futures)

    async def _sparse_tensors_prune_small(self, epsilon):
        futures = []
        for tensor in self._embedding_operators:
            if not tensor.is_backing:
                future = tensor._sparse_tensor_prune_small(epsilon)
                futures.append(future)
        await asyncio.gather(*futures)

    async def _sparse_tensors_prune_old(self, max_age):
        futures = []
        for tensor in self._embedding_operators:
            if not tensor.is_backing:
                future = tensor._sparse_tensor_prune_old(max_age)
                futures.append(future)
        await asyncio.gather(*futures)

    def _do_export(self, path, *, model_export_selector=None, output_names=None):
        asyncio.run(self._sparse_tensors_export(
            path, model_export_selector=model_export_selector))
        super()._do_export(path, model_export_selector=model_export_selector, output_names=output_names)

    def _get_export_meta(self, path, *, model_export_selector=None):
        sparse_data_dir = os.path.basename(path) + '.msd'
        sparse_tensors = []
        module = self.module
        name_prefix = None
        if model_export_selector is not None:
            func, name_prefix = model_export_selector
            module = func(module)
        selected = set(module.modules())
        for tensor in self._embedding_operators:
            if tensor.item in selected and tensor.is_exported:
                name = tensor.name
                if name_prefix is not None:
                    if not name.startswith(name_prefix):
                        message = f"tensor name {name!r} mismatches with name prefix {name_prefix!r}"
                        raise RuntimeError(message)
                    name = name[len(name_prefix):]
                data_dir = name + '.msm'
                partition_count = tensor._handle.partition_count
                sparse_tensor = {
                    'name': name,
                    'data_dir': data_dir,
                    'partition_count': partition_count,
                }
                sparse_tensors.append(sparse_tensor)
        meta = super()._get_export_meta(path, model_export_selector=model_export_selector)
        meta['sparse_data_dir'] = sparse_data_dir
        meta['sparse_tensors'] = sparse_tensors
        return meta

    def _do_prune_small(self, epsilon):
        self.agent.barrier()
        if self.agent.rank == 0:
            asyncio.run(self._sparse_tensors_prune_small(epsilon))
        self.agent.barrier()

    def _do_prune_old(self, max_age):
        self.agent.barrier()
        if self.agent.rank == 0:
            asyncio.run(self._sparse_tensors_prune_old(max_age))
        self.agent.barrier()

    def _execute_combine(self, minibatch):
        for tensor in self._embedding_operators:
            if not tensor.is_backing:
                tensor.item._combine(minibatch)

    def _execute_pull(self):
        asyncio.run(self._pull_tensors())

    def _execute_compute(self):
        for tensor in self._embedding_operators:
            if not tensor.is_backing:
                tensor.item._compute()

    def _execute_cast(self, minibatch):
        for mod in self._cast_operators:
            mod._cast(minibatch)

    def __call__(self, minibatch):
        self._execute_combine(minibatch)
        self._execute_pull()
        self._execute_compute()
        self._execute_cast(minibatch)
        fake_input = torch.tensor(0.0)
        x = self.module(fake_input)
        return x
