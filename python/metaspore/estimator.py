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

import io
import torch
import pyspark.ml.base
from . import patching_pickle
from .agent import Agent
from .model import Model
from .updater import TensorUpdater
from .updater import AdamTensorUpdater
from .distributed_trainer import DistributedTrainer
from .file_utils import dir_exists
from .file_utils import delete_dir
from .ps_launcher import PSLauncher

class PyTorchAgent(Agent):
    def __init__(self):
        super().__init__()
        self.module = None
        self.updater = None
        self.dataset = None
        self.model_export_selector = None
        self.tensor_name_prefix = None
        self.model = None
        self.trainer = None
        self.is_training_mode = None
        self.validation_result = None
        self.model_in_path = None
        self.model_out_path = None
        self.model_export_path = None
        self.model_version = None
        self.model_output_names = None
        self.experiment_name = None
        self.use_fresh_updaters = None
        self.training_epoches = None
        self.shuffle_training_dataset = None
        self.max_sparse_feature_age = None
        self.metric_update_interval = None
        self.consul_host = None
        self.consul_port = None
        self.consul_endpoint_prefix = None
        self.consul_model_sync_command = None
        self.input_label_column_index = None
        self.output_label_column_name = None
        self.output_label_column_type = None
        self.output_prediction_column_name = None
        self.output_prediction_column_type = None
        self.minibatch_id = 0

    def run(self):
        self.distribute_module()
        self.distribute_updater()
        self.start_workers()
        self.feed_dataset()
        self.collect_module()
        self.stop_workers()

    def distribute_module(self):
        buf = io.BytesIO()
        self._save_custom_initializer_and_updaters()
        torch.save(self.module, buf, pickle_module=patching_pickle)
        self._restore_custom_initializer_and_updaters()
        module = buf.getvalue()
        model_export_selector = self.model_export_selector
        rdd = self.spark_context.parallelize(range(self.worker_count), self.worker_count)
        rdd.barrier().mapPartitions(lambda _: __class__._distribute_module(module, model_export_selector, _)).collect()

    @classmethod
    def _distribute_module(cls, module, model_export_selector, _):
        buf = io.BytesIO(module)
        module = torch.load(buf)
        self = __class__.get_instance()
        self.module = module
        self.model_export_selector = model_export_selector
        self._restore_custom_initializer_and_updaters()
        return _

    def _is_batch_norm(self, name, mod):
        if isinstance(mod, torch.nn.modules.batchnorm._BatchNorm):
            if isinstance(mod, torch.nn.SyncBatchNorm):
                message = f"module {name!r} is an instance of torch.nn.SyncBatchNorm, "
                message += "which is not supported by DistributedTrainer"
                raise RuntimeError(message)
            return True
        return False

    def _save_custom_initializer_and_updaters(self):
        for name, mod in self.module.named_modules():
            if self._is_batch_norm(name, mod):
                self._save_custom_initializer_and_updater(name, mod)

    def _save_custom_initializer_and_updater(self, name, mod):
        if mod.running_mean is not None and getattr(mod.running_mean, 'initializer', None) is not None:
            mod._running_mean_initializer = mod.running_mean.initializer
            del mod.running_mean.initializer
        if mod.running_mean is not None and getattr(mod.running_mean, 'updater', None) is not None:
            mod._running_mean_updater = mod.running_mean.updater
            del mod.running_mean.updater
        if mod.running_var is not None and getattr(mod.running_var, 'initializer', None) is not None:
            mod._running_var_initializer = mod.running_var.initializer
            del mod.running_var.initializer
        if mod.running_var is not None and getattr(mod.running_var, 'updater', None) is not None:
            mod._running_var_updater = mod.running_var.updater
            del mod.running_var.updater

    def _restore_custom_initializer_and_updaters(self):
        for name, mod in self.module.named_modules():
            if self._is_batch_norm(name, mod):
                self._restore_custom_initializer_and_updater(name, mod)

    def _restore_custom_initializer_and_updater(self, name, mod):
        if mod.running_mean is not None and getattr(mod, '_running_mean_initializer', None) is not None:
            mod.running_mean.initializer = mod._running_mean_initializer
            del mod._running_mean_initializer
        if mod.running_mean is not None and getattr(mod, '_running_mean_updater', None) is not None:
            mod.running_mean.updater = mod._running_mean_updater
            del mod._running_mean_updater
        if mod.running_var is not None and getattr(mod, '_running_var_initializer', None) is not None:
            mod.running_var.initializer = mod._running_var_initializer
            del mod._running_var_initializer
        if mod.running_var is not None and getattr(mod, '_running_var_updater', None) is not None:
            mod.running_var.updater = mod._running_var_updater
            del mod._running_var_updater

    def distribute_updater(self):
        updater = self.updater
        rdd = self.spark_context.parallelize(range(self.worker_count), self.worker_count)
        rdd.barrier().mapPartitions(lambda _: __class__._distribute_updater(updater, _)).collect()

    @classmethod
    def _distribute_updater(cls, updater, _):
        self = __class__.get_instance()
        self.updater = updater
        return _

    def collect_module(self):
        if self.is_training_mode:
            rdd = self.spark_context.parallelize(range(self.worker_count), self.worker_count)
            state, = rdd.barrier().mapPartitions(lambda _: __class__._collect_module(_)).collect()
            self.module.load_state_dict(state)

    @classmethod
    def _collect_module(cls, _):
        self = __class__.get_instance()
        self.model.sync()
        if self.rank != 0:
            return ()
        state = self.module.state_dict()
        return state,

    def setup_model(self):
        self.model = Model.wrap(self, self.module, name_prefix=self.tensor_name_prefix)

    def setup_trainer(self):
        self.trainer = DistributedTrainer(self.model, updater=self.updater)
        self.trainer.initialize()

    def worker_start(self):
        self.setup_model()
        self.setup_trainer()
        self.load_model()

    def load_model(self):
        if self.model_in_path is not None:
            keep_meta = self.use_fresh_updaters
            print('\033[38;5;196mloading model from %s (keep_meta=%s)\033[m' %
                  (self.model_in_path, keep_meta))
            self.trainer.load(self.model_in_path, keep_meta=keep_meta)

    def save_model(self):
        self.model.prune_old(self.max_sparse_feature_age)
        if self.model_out_path is not None:
            print('\033[38;5;196msaving model to %s\033[m' % self.model_out_path)
            self.trainer.save(self.model_out_path)

    def export_model(self):
        if self.model_export_path is not None:
            print('\033[38;5;196mexporting model to %s\033[m' % self.model_export_path)
            self.model.eval()
            self.model.model_version = self.model_version
            self.model.experiment_name = self.experiment_name
            self.model.prune_small(0.0)
            self.model.export(self.model_export_path,
                              model_export_selector=self.model_export_selector,
                              output_names=self.model_output_names)

    def worker_stop(self):
        # Make sure the final metric buffers are pushed.
        self.push_metric()
        if self.is_training_mode:
            self.save_model()
            self.export_model()

    def feed_dataset(self):
        if self.is_training_mode:
            self.feed_training_dataset()
        else:
            self.feed_validation_dataset()

    def feed_training_dataset(self):
        from .input import shuffle_df
        for epoch in range(self.training_epoches):
            df = self.dataset
            if self.shuffle_training_dataset:
                df = shuffle_df(df, self.worker_count)
            df = df.select(self.feed_training_minibatch()(*df.columns).alias('train'))
            df.write.format('noop').mode('overwrite').save()

    def feed_validation_dataset(self):
        df = self.dataset.withColumn(self.output_prediction_column_name,
                                     self.feed_validation_minibatch()(*self.dataset.columns))
        df = df.withColumn(self.output_label_column_name,
                           df[self.input_label_column_index].cast(self.output_label_column_type))
        df = df.withColumn(self.output_prediction_column_name,
                           df[self.output_prediction_column_name].cast(self.output_prediction_column_type))
        self.validation_result = df
        # PySpark DataFrame & RDD is lazily evaluated.
        # We must call ``cache`` here otherwise PySpark will try to reevaluate
        # ``validation_result`` when we use it, which is not possible as the
        # PS system has been shutdown.
        df.cache()
        df.write.format('noop').mode('overwrite').save()

    def feed_training_minibatch(self):
        from pyspark.sql.types import FloatType
        from pyspark.sql.functions import pandas_udf
        @pandas_udf(returnType=FloatType())
        def _feed_training_minibatch(*minibatch):
            self = __class__.get_instance()
            result = self.train_minibatch(minibatch)
            result = self.process_minibatch_result(minibatch, result)
            return result
        return _feed_training_minibatch

    def feed_validation_minibatch(self):
        from pyspark.sql.types import FloatType
        from pyspark.sql.functions import pandas_udf
        @pandas_udf(returnType=FloatType())
        def _feed_validation_minibatch(*minibatch):
            self = __class__.get_instance()
            result = self.validate_minibatch(minibatch)
            result = self.process_minibatch_result(minibatch, result)
            return result
        return _feed_validation_minibatch

    def preprocess_minibatch(self, minibatch):
        import numpy as np
        ndarrays = [col.values for col in minibatch]
        labels = minibatch[self.input_label_column_index].values.astype(np.float32)
        return ndarrays, labels

    def process_minibatch_result(self, minibatch, result):
        import pandas as pd
        minibatch_size = len(minibatch[self.input_label_column_index])
        if result is None:
            result = pd.Series([0.0] * minibatch_size)
        if len(result) != minibatch_size:
            message = "result length (%d) and " % len(result)
            message += "minibatch size (%d) mismatch" % minibatch_size
            raise RuntimeError(message)
        if not isinstance(result, pd.Series):
            if len(result.reshape(-1)) == minibatch_size:
                result = result.reshape(-1)
            else:
                message = "result can not be converted to pandas series; "
                message += "result.shape: {}, ".format(result.shape)
                message += "minibatch_size: {}".format(minibatch_size)
                raise RuntimeError(message)
            result = pd.Series(result)
        return result

    def train_minibatch(self, minibatch):
        self.model.train()
        ndarrays, labels = self.preprocess_minibatch(minibatch)
        predictions = self.model(ndarrays)
        labels = torch.from_numpy(labels).reshape(-1, 1)
        loss = self.compute_loss(predictions, labels)
        self.trainer.train(loss)
        self.update_progress(predictions, labels)

    def validate_minibatch(self, minibatch):
        self.model.eval()
        ndarrays, labels = self.preprocess_minibatch(minibatch)
        predictions = self.model(ndarrays)
        labels = torch.from_numpy(labels).reshape(-1, 1)
        self.update_progress(predictions, labels)
        return predictions.detach().reshape(-1)

    def compute_loss(self, predictions, labels):
        from .loss_utils import log_loss
        return log_loss(predictions, labels) / labels.shape[0]

    def update_progress(self, predictions, labels):
        self.minibatch_id += 1
        self.update_metric(predictions, labels)
        if self.minibatch_id % self.metric_update_interval == 0:
            self.push_metric()

class PyTorchLauncher(PSLauncher):
    def __init__(self):
        super().__init__()
        self.module = None
        self.updater = None
        self.dataset = None
        self.model_export_selector = None
        self.tensor_name_prefix = None
        self.worker_count = None
        self.server_count = None
        self.agent_class = None
        self.agent_object = None
        self.is_training_mode = None
        self.model_in_path = None
        self.model_out_path = None
        self.model_export_path = None
        self.model_version = None
        self.model_output_names = None
        self.experiment_name = None
        self.use_fresh_updaters = None
        self.training_epoches = None
        self.shuffle_training_dataset = None
        self.max_sparse_feature_age = None
        self.metric_update_interval = None
        self.consul_host = None
        self.consul_port = None
        self.consul_endpoint_prefix = None
        self.consul_model_sync_command = None
        self.input_label_column_index = None
        self.output_label_column_name = None
        self.output_label_column_type = None
        self.output_prediction_column_name = None
        self.output_prediction_column_type = None
        self.extra_agent_attributes = None

    def _get_agent_class(self):
        return self.agent_class

    def _initialize_agent(self, agent):
        agent.module = self.module
        agent.updater = self.updater
        agent.dataset = self.dataset
        agent.model_export_selector = self.model_export_selector
        self.agent_object = agent

    def launch(self):
        self._worker_count = self.worker_count
        self._server_count = self.server_count
        self._agent_attributes = dict()
        self._agent_attributes['tensor_name_prefix'] = self.tensor_name_prefix
        self._agent_attributes['is_training_mode'] = self.is_training_mode
        self._agent_attributes['model_in_path'] = self.model_in_path
        self._agent_attributes['model_out_path'] = self.model_out_path
        self._agent_attributes['model_export_path'] = self.model_export_path
        self._agent_attributes['model_version'] = self.model_version
        self._agent_attributes['model_output_names'] = self.model_output_names
        self._agent_attributes['experiment_name'] = self.experiment_name
        self._agent_attributes['use_fresh_updaters'] = self.use_fresh_updaters
        self._agent_attributes['training_epoches'] = self.training_epoches
        self._agent_attributes['shuffle_training_dataset'] = self.shuffle_training_dataset
        self._agent_attributes['max_sparse_feature_age'] = self.max_sparse_feature_age
        self._agent_attributes['metric_update_interval'] = self.metric_update_interval
        self._agent_attributes['consul_host'] = self.consul_host
        self._agent_attributes['consul_port'] = self.consul_port
        self._agent_attributes['consul_endpoint_prefix'] = self.consul_endpoint_prefix
        self._agent_attributes['consul_model_sync_command'] = self.consul_model_sync_command
        self._agent_attributes['input_label_column_index'] = self.input_label_column_index
        self._agent_attributes['output_label_column_name'] = self.output_label_column_name
        self._agent_attributes['output_label_column_type'] = self.output_label_column_type
        self._agent_attributes['output_prediction_column_name'] = self.output_prediction_column_name
        self._agent_attributes['output_prediction_column_type'] = self.output_prediction_column_type
        self._agent_attributes.update(self.extra_agent_attributes)
        self._keep_session = True
        self.launch_agent()

class PyTorchHelperMixin(object):
    def __init__(self,
                 module=None,
                 updater=None,
                 worker_count=100,
                 server_count=100,
                 agent_class=None,
                 model_in_path=None,
                 model_out_path=None,
                 model_export_path=None,
                 model_version=None,
                 model_output_names=None,
                 use_fresh_updaters=True,
                 experiment_name=None,
                 training_epoches=1,
                 shuffle_training_dataset=False,
                 max_sparse_feature_age=15,
                 metric_update_interval=10,
                 consul_host=None,
                 consul_port=None,
                 consul_endpoint_prefix=None,
                 consul_model_sync_command=None,
                 input_label_column_index=1,
                 output_label_column_name='label',
                 output_label_column_type='double',
                 output_prediction_column_name='rawPrediction',
                 output_prediction_column_type='double',
                 **kwargs):
        super().__init__()
        self.module = module
        self.updater = updater
        self.worker_count = worker_count
        self.server_count = server_count
        self.agent_class = agent_class
        self.model_in_path = model_in_path
        self.model_out_path = model_out_path
        self.model_export_path = model_export_path
        self.model_version = model_version
        self.model_output_names = model_output_names
        self.experiment_name = experiment_name
        self.use_fresh_updaters = use_fresh_updaters
        self.training_epoches = training_epoches
        self.shuffle_training_dataset = shuffle_training_dataset
        self.max_sparse_feature_age = max_sparse_feature_age
        self.metric_update_interval = metric_update_interval
        self.consul_host = consul_host
        self.consul_port = consul_port
        self.consul_endpoint_prefix = consul_endpoint_prefix
        self.consul_model_sync_command = consul_model_sync_command
        self.input_label_column_index = input_label_column_index
        self.output_label_column_name = output_label_column_name
        self.output_label_column_type = output_label_column_type
        self.output_prediction_column_name = output_prediction_column_name
        self.output_prediction_column_type = output_prediction_column_type
        self.extra_agent_attributes = kwargs
        self.final_metric = None

    def _check_properties(self):
        if not isinstance(self.module, torch.nn.Module):
            raise TypeError(f"module must be torch.nn.Module; {self.module!r} is invalid")
        if self.updater is not None and not isinstance(self.updater, TensorUpdater):
            raise TypeError(f"updater must be TensorUpdater; {self.updater!r} is invalid")
        if not isinstance(self.worker_count, int) or self.worker_count <= 0:
            raise TypeError(f"worker_count must be positive integer; {self.worker_count!r} is invalid")
        if not isinstance(self.server_count, int) or self.server_count <= 0:
            raise TypeError(f"server_count must be positive integer; {self.server_count!r} is invalid")
        if self.agent_class is not None and not issubclass(self.agent_class, PyTorchAgent):
            raise TypeError(f"agent_class must be subclass of PyTorchAgent; {self.agent_class!r} is invalid")
        if self.model_in_path is not None and not isinstance(self.model_in_path, str):
            raise TypeError(f"model_in_path must be string; {self.model_in_path!r} is invalid")
        if self.model_in_path is not None and not self.model_in_path.endswith('/'):
            self.model_in_path += '/'
        if self.model_in_path is not None and not dir_exists(self.model_in_path):
            raise RuntimeError(f"model_in_path {self.model_in_path!r} does not exist")
        if self.model_out_path is not None and not isinstance(self.model_out_path, str):
            raise TypeError(f"model_out_path must be string; {self.model_out_path!r} is invalid")
        if self.model_out_path is not None and not self.model_out_path.endswith('/'):
            self.model_out_path += '/'
        if self.model_export_path is not None and not isinstance(self.model_export_path, str):
            raise TypeError(f"model_export_path must be string; {self.model_export_path!r} is invalid")
        if self.model_export_path is not None and not self.model_export_path.endswith('/'):
            self.model_export_path += '/'
        if self.model_version is not None and not isinstance(self.model_version, str):
            raise TypeError(f"model_version must be string; {self.model_version!r} is invalid")
        if self.model_output_names is not None and not isinstance(self.model_output_names, (list, tuple)):
            raise TypeError(f"model_output_names must be list or tuple; {self.model_output_names!r} is invalid")
        if self.model_output_names is not None and not all(isinstance(item, str) for item in self.model_output_names):
            raise TypeError(f"model_output_names must be list or tuple of string; {self.model_output_names!r} is invalid")
        if self.experiment_name is not None and not isinstance(self.experiment_name, str):
            raise TypeError(f"experiment_name must be string; {self.experiment_name!r} is invalid")
        if not isinstance(self.training_epoches, int) or self.training_epoches <= 0:
            raise TypeError(f"training_epoches must be positive integer; {self.training_epoches!r} is invalid")
        if not isinstance(self.max_sparse_feature_age, int) or self.max_sparse_feature_age <= 0:
            raise TypeError(f"max_sparse_feature_age must be positive integer; {self.max_sparse_feature_age!r} is invalid")
        if not isinstance(self.metric_update_interval, int) or self.metric_update_interval <= 0:
            raise TypeError(f"metric_update_interval must be positive integer; {self.metric_update_interval!r} is invalid")
        if self.consul_host is not None and not isinstance(self.consul_host, str):
            raise TypeError(f"consul_host must be string; {self.consul_host!r} is invalid")
        if self.consul_port is not None and (not isinstance(self.consul_port, int) or self.consul_port <= 0):
            raise TypeError(f"consul_port must be positive integer; {self.consul_port!r} is invalid")
        if self.consul_endpoint_prefix is not None and not isinstance(self.consul_endpoint_prefix, str):
            raise TypeError(f"consul_endpoint_prefix must be string; {self.consul_endpoint_prefix!r} is invalid")
        if self.consul_endpoint_prefix is not None and not self.consul_endpoint_prefix.strip('/'):
            raise ValueError(f"consul_endpoint_prefix {self.consul_endpoint_prefix!r} is invalid")
        if self.consul_endpoint_prefix is not None:
            self.consul_endpoint_prefix = self.consul_endpoint_prefix.strip('/')
        if self.consul_model_sync_command is not None and not isinstance(self.consul_model_sync_command, str):
            raise TypeError(f"consul_model_sync_command must be string; {self.consul_model_sync_command!r} is invalid")
        if not isinstance(self.input_label_column_index, int) or self.input_label_column_index < 0:
            raise TypeError(f"input_label_column_index must be non-negative integer; {self.input_label_column_index!r} is invalid")
        if not isinstance(self.output_label_column_name, str):
            raise TypeError(f"output_label_column_name must be string; {self.output_label_column_name!r} is invalid")
        if not isinstance(self.output_label_column_type, str):
            raise TypeError(f"output_label_column_type must be string; {self.output_label_column_type!r} is invalid")
        if not isinstance(self.output_prediction_column_name, str):
            raise TypeError(f"output_prediction_column_name must be string; {self.output_prediction_column_name!r} is invalid")
        if not isinstance(self.output_prediction_column_type, str):
            raise TypeError(f"output_prediction_column_type must be string; {self.output_prediction_column_type!r} is invalid")
        if self.model_export_path is not None and (self.model_version is None or self.experiment_name is None):
            raise RuntimeError("model_version and experiment_name are required when model_export_path is specified")
        if self.consul_endpoint_prefix is not None and (self.consul_host is None or self.consul_port is None):
            raise RuntimeError("consul_host and consul_port are required when consul_endpoint_prefix is specified")
        if self.consul_endpoint_prefix is not None and self.model_export_path is None:
            raise RuntimeError("model_export_path is required when consul_endpoint_prefix is specified")

    def _get_launcher_class(self):
        return PyTorchLauncher

    def _get_model_class(self):
        return PyTorchModel

    def _get_agent_class(self):
        return self.agent_class or PyTorchAgent

    def _get_updater_object(self):
        return self.updater or AdamTensorUpdater(1e-5)

    def _create_launcher(self, dataset, is_training_mode):
        self._check_properties()
        launcher = self._get_launcher_class()()
        launcher.module = self.module
        launcher.updater = self._get_updater_object()
        launcher.dataset = dataset
        launcher.worker_count = self.worker_count
        launcher.server_count = self.server_count
        launcher.agent_class = self._get_agent_class()
        launcher.is_training_mode = is_training_mode
        launcher.model_in_path = self.model_in_path
        launcher.model_out_path = self.model_out_path
        launcher.model_export_path = self.model_export_path
        launcher.model_version = self.model_version
        launcher.model_output_names = self.model_output_names
        launcher.experiment_name = self.experiment_name
        launcher.use_fresh_updaters = self.use_fresh_updaters
        launcher.training_epoches = self.training_epoches
        launcher.shuffle_training_dataset = self.shuffle_training_dataset
        launcher.max_sparse_feature_age = self.max_sparse_feature_age
        launcher.metric_update_interval = self.metric_update_interval
        launcher.consul_host = self.consul_host
        launcher.consul_port = self.consul_port
        launcher.consul_endpoint_prefix = self.consul_endpoint_prefix
        launcher.consul_model_sync_command = self.consul_model_sync_command
        launcher.input_label_column_index = self.input_label_column_index
        launcher.output_label_column_name = self.output_label_column_name
        launcher.output_label_column_type = self.output_label_column_type
        launcher.output_prediction_column_name = self.output_prediction_column_name
        launcher.output_prediction_column_type = self.output_prediction_column_type
        launcher.extra_agent_attributes = self.extra_agent_attributes
        return launcher

    def _get_model_arguments(self, module):
        args = self.extra_agent_attributes.copy()
        args['module'] = module
        args['updater'] = self.updater
        args['worker_count'] = self.worker_count
        args['server_count'] = self.server_count
        args['agent_class'] = self.agent_class
        args['model_in_path'] = self.model_out_path
        args['model_export_path'] = self.model_export_path
        args['model_version'] = self.model_version
        args['model_output_names'] = self.model_output_names
        args['experiment_name'] = self.experiment_name
        args['metric_update_interval'] = self.metric_update_interval
        args['consul_host'] = self.consul_host
        args['consul_port'] = self.consul_port
        args['consul_endpoint_prefix'] = self.consul_endpoint_prefix
        args['consul_model_sync_command'] = self.consul_model_sync_command
        args['input_label_column_index'] = self.input_label_column_index
        args['output_label_column_name'] = self.output_label_column_name
        args['output_label_column_type'] = self.output_label_column_type
        args['output_prediction_column_name'] = self.output_prediction_column_name
        args['output_prediction_column_type'] = self.output_prediction_column_type
        return args

    def _create_model(self, module):
        args = self._get_model_arguments(module)
        model = self._get_model_class()(**args)
        return model

class PyTorchModel(PyTorchHelperMixin, pyspark.ml.base.Model):
    def _transform(self, dataset):
        launcher = self._create_launcher(dataset, False)
        launcher.launch()
        result = launcher.agent_object.validation_result
        self.final_metric = launcher.agent_object._metric
        return result

    def publish(self):
        import json
        import consul
        if not self.consul_endpoint_prefix:
            message = f"consul_endpoint_prefix is not specified; can not publish {self.experiment_name!r}"
            raise RuntimeError(message)
        util_cmd = self.consul_model_sync_command
        if util_cmd is None:
            util_cmd = 'aws s3 cp --recursive'
        data = {
            'name': self.experiment_name,
            'service': self.experiment_name + '-service',
            'path': self.model_export_path,
            'version': self.model_version,
            'util_cmd': util_cmd,
        }
        string = json.dumps(data, separators=(',', ': '), indent=4)
        consul_client = consul.Consul(host=self.consul_host, port=self.consul_port)
        index, response = consul_client.kv.get(self.consul_endpoint_prefix, keys=True)
        if response is None:
            consul_client.kv.put(self.consul_endpoint_prefix + '/', value=None)
            print("init consul endpoint dir: %s" % self.consul_endpoint_prefix)
        endpoint = '%s/%s' % (self.consul_endpoint_prefix, self.experiment_name)
        consul_path = f'{self.consul_host}:{self.consul_port}/{endpoint}'
        try:
            consul_client.kv.put(endpoint, string)
            message = f"notify consul succeed: {consul_path}"
            print(message)
        except consul.Timeout as err:
            raise TimeoutError(f"notify consul {consul_path} timeout err: {err}") from err

class PyTorchEstimator(PyTorchHelperMixin, pyspark.ml.base.Estimator):
    def _check_properties(self):
        super()._check_properties()
        if self.model_out_path is None:
            # ``model_out_path`` must be specified otherwise an instance of PyTorchModel
            # does not known where to load the model from. This is due to the approach
            # we implement ``_fit`` and ``_transform`` that the PS system will be
            # shutdown when ``_fit`` finishes and restarted in ``_transform``.
            # Later, we may refine the implementation of PS to remove this limitation.
            raise RuntimeError("model_out_path of estimator must be specified")

    def _clear_output(self):
        import os
        if self.model_out_path is not None:
            delete_dir(os.path.join(self.model_out_path, self.experiment_name))
        if self.model_export_path is not None:
            delete_dir(os.path.join(self.model_export_path, self.experiment_name))

    def _fit(self, dataset):
        self._clear_output()
        launcher = self._create_launcher(dataset, True)
        launcher.launch()
        module = launcher.agent_object.module
        module.eval()
        model = self._create_model(module)
        self.final_metric = launcher.agent_object._metric
        return model
