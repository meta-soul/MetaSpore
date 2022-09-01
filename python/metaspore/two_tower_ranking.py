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

import asyncio
import torch
import pyspark
from .embedding import EmbeddingOperator
from .estimator import PyTorchAgent
from .estimator import PyTorchLauncher
from .estimator import PyTorchModel
from .estimator import PyTorchEstimator

class TwoTowerRankingModule(torch.nn.Module):
    def __init__(self, user_module, item_module, item_embedding_module, similarity_module):
        super().__init__()
        if not isinstance(user_module, torch.nn.Module):
            raise TypeError(f"user_module must be torch.nn.Module; {user_module!r} is invalid")
        if not isinstance(item_module, torch.nn.Module):
            raise TypeError(f"item_module must be torch.nn.Module; {item_module!r} is invalid")
        if not isinstance(item_embedding_module, torch.nn.Module):
            raise TypeError(f"item_embedding_module must be torch.nn.Module; {item_embedding_module!r} is invalid")
        if not isinstance(similarity_module, torch.nn.Module):
            raise TypeError(f"similarity_module must be torch.nn.Module; {similarity_module!r} is invalid")
        self._user_module = user_module
        self._item_module = item_module
        self._item_embedding_module = item_embedding_module
        self._similarity_module = similarity_module

    @property
    def user_module(self):
        return self._user_module

    @property
    def item_module(self):
        return self._item_module

    @property
    def item_embedding_module(self):
        return self._item_embedding_module

    @property
    def similarity_module(self):
        return self._similarity_module

    def _get_item_embedding(self, x):
        if self.training or self._item_embedding_module is None:
            if self._item_module is None:
                raise RuntimeError("item_module is None")
            item_emb = self._item_module(x)
        else:
            if self._item_embedding_module is None:
                raise RuntimeError("item_embedding_module is None")
            item_emb = self._item_embedding_module(x)
        return item_emb

    def forward(self, x):
        user_emb = self._user_module(x)
        item_emb = self._get_item_embedding(x)
        sim = self._similarity_module(user_emb, item_emb)
        return sim

class TwoTowerRankingAgent(PyTorchAgent):
    def _mark_unexported_operators(self, module):
        for name, mod in module.named_modules():
            if isinstance(mod, EmbeddingOperator):
                mod.is_exported = False

    def _unmark_unexported_operators(self, module):
        for name, mod in module.named_modules():
            if isinstance(mod, EmbeddingOperator):
                mod.is_exported = True

    def _mark_backing_operators(self, module):
        for name, mod in module.named_modules():
            if isinstance(mod, EmbeddingOperator):
                mod.is_backing = True

    def _unmark_backing_operators(self, module):
        for name, mod in module.named_modules():
            if isinstance(mod, EmbeddingOperator):
                mod.is_backing = False

    ## train

    @classmethod
    def _handle_item_embedding_module_for_train(cls, _):
        self = __class__.get_instance()
        self._mark_backing_operators(self.module.item_embedding_module)
        return _

    @classmethod
    def _restore_handle_item_embedding_module_for_train(cls, _):
        self = __class__.get_instance()
        self._unmark_backing_operators(self.module.item_embedding_module)
        return _

    def feed_training_dataset(self):
        rdd = self.spark_context.parallelize(range(self.worker_count), self.worker_count)
        rdd.barrier().mapPartitions(self._handle_item_embedding_module_for_train).collect()
        super().feed_training_dataset()
        rdd = self.spark_context.parallelize(range(self.worker_count), self.worker_count)
        rdd.barrier().mapPartitions(self._restore_handle_item_embedding_module_for_train).collect()
        self.feed_item_dataset()

    ## item_predict

    @classmethod
    def _pull_model_for_item_predict(cls, _):
        self = __class__.get_instance()
        asyncio.run(self.model._pull_tensors(force_mode=True))
        return _

    @classmethod
    def _handle_submodel_for_item_predict(cls, _):
        self = __class__.get_instance()
        self._item_predict_submodel = self.model.get_submodel(self.module.item_module, '_item_module.')
        return _

    @classmethod
    def _restore_handle_submodel_for_item_predict(cls, _):
        self = __class__.get_instance()
        del self._item_predict_submodel
        return _

    def feed_item_dataset(self):
        if self.item_dataset is not None:
            rdd = self.spark_context.parallelize(range(self.worker_count), self.worker_count)
            rdd.barrier().mapPartitions(self._pull_model_for_item_predict).collect()
            rdd = self.spark_context.parallelize(range(self.worker_count), self.worker_count)
            rdd.barrier().mapPartitions(self._handle_submodel_for_item_predict).collect()
            df = self.item_dataset
            func = self.feed_item_minibatch()
            df = df.mapInPandas(func, df.schema)
            df.write.format('noop').mode('overwrite').save()
            rdd = self.spark_context.parallelize(range(self.worker_count), self.worker_count)
            rdd.barrier().mapPartitions(self._restore_handle_submodel_for_item_predict).collect()

    ## online_predict

    def _handle_item_module_for_online_predict(self):
        self._mark_unexported_operators(self.module.item_module)
        self._saved_item_module = self.module._item_module
        self.module._item_module = None

    def _restore_handle_item_module_for_online_predict(self):
        self.module._item_module = self._saved_item_module
        del self._saved_item_module
        self._unmark_unexported_operators(self.module.item_module)

    def export_model(self):
        if self.model_export_path is not None:
            self._handle_item_module_for_online_predict()
            super().export_model()
            self._restore_handle_item_module_for_online_predict()

    ## offline_predict

    def _handle_module_for_offline_predict(self):
        if self.use_amended_module_for_offline_predict:
            self._saved_item_module = self.module._item_module
            self.module._item_module = None
        else:
            self._saved_item_embedding_module = self.module._item_embedding_module
            self.module._item_embedding_module = None

    def _restore_handle_module_for_offline_predict(self):
        if self.use_amended_module_for_offline_predict:
            self.module._item_module = self._saved_item_module
            del self._saved_item_module
        else:
            self.module._item_embedding_module = self._saved_item_embedding_module
            del self._saved_item_embedding_module

    def setup_trainer(self):
        if not self.is_training_mode:
            self._handle_module_for_offline_predict()
        super().setup_trainer()

    def worker_stop(self):
        if not self.is_training_mode:
            self._restore_handle_module_for_offline_predict()
        super().worker_stop()

    ## helper methods for item_predict

    def feed_item_minibatch(self):
        def _feed_item_minibatch(iterator):
            self = __class__.get_instance()
            for minibatch in iterator:
                self.predict_item_minibatch(minibatch)
                yield minibatch
        return _feed_item_minibatch

    def _execute_combine(self, module, minibatch):
        for name, mod in module.named_modules():
            if isinstance(mod, EmbeddingOperator):
                indices, offsets = mod._do_combine(minibatch)
                return indices

    def _execute_push(self, module, keys, embeddings):
        for name, mod in module.named_modules():
            if isinstance(mod, EmbeddingOperator):
                mod.keys_and_data = keys, embeddings
                tensor = mod._distributed_tensor
                asyncio.run(tensor._push_tensor(is_value=True))
                mod._clean()
                return

    def _execute_item_embedding_combine(self, minibatch):
        keys = self._execute_combine(self.module.item_embedding_module, minibatch)
        return keys

    def _execute_item_embedding_push(self, keys, embeddings):
        self._execute_push(self.module.item_embedding_module, keys, embeddings)

    def predict_item_minibatch(self, minibatch):
        self._item_predict_submodel.eval()
        predictions = self._item_predict_submodel(minibatch)
        embeddings = predictions.detach().numpy()
        keys = self._execute_item_embedding_combine(minibatch)
        self._execute_item_embedding_push(keys, embeddings)

class TwoTowerRankingLauncher(PyTorchLauncher):
    def __init__(self):
        super().__init__()
        self.item_dataset = None

    def _initialize_agent(self, agent):
        agent.item_dataset = self.item_dataset
        super()._initialize_agent(agent)

class TwoTowerRankingHelperMixin(object):
    def __init__(self, item_dataset=None, use_amended_module_for_offline_predict=False, **kwargs):
        super().__init__(**kwargs)
        self.item_dataset = item_dataset
        self.use_amended_module_for_offline_predict = use_amended_module_for_offline_predict
        self.extra_agent_attributes['use_amended_module_for_offline_predict'] = self.use_amended_module_for_offline_predict

    def _check_properties(self):
        super()._check_properties()
        if not isinstance(self.module, TwoTowerRankingModule):
            raise TypeError(f"module must be TwoTowerRankingModule; {self.module!r} is invalid")
        if self.item_dataset is not None and not isinstance(self.item_dataset, pyspark.sql.DataFrame):
            raise TypeError(f"item_dataset must be pyspark.sql.DataFrame; {self.item_dataset!r} is invalid")

    def _get_launcher_class(self):
        return TwoTowerRankingLauncher

    def _get_model_class(self):
        return TwoTowerRankingModel

    def _get_agent_class(self):
        return self.agent_class or TwoTowerRankingAgent

    def _get_model_arguments(self, module):
        args = super()._get_model_arguments(module)
        args['item_dataset'] = self.item_dataset
        args['use_amended_module_for_offline_predict'] = self.use_amended_module_for_offline_predict
        return args

    def _create_launcher(self, dataset, is_training_mode):
        launcher = super()._create_launcher(dataset, is_training_mode)
        if is_training_mode:
            launcher.item_dataset = self.item_dataset
        return launcher

class TwoTowerRankingModel(TwoTowerRankingHelperMixin, PyTorchModel):
    pass

class TwoTowerRankingEstimator(TwoTowerRankingHelperMixin, PyTorchEstimator):
    def _check_properties(self):
        super()._check_properties()
        if self.model_export_path is not None and self.item_dataset is None:
            raise RuntimeError("item_dataset must be specified to export model")
        if self.use_amended_module_for_offline_predict and self.item_dataset is None:
            raise RuntimeError("item_dataset must be specified to use amended module for offline predict")
