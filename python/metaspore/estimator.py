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
from .metric import ModelMetric
from .metric import BinaryClassificationModelMetric
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
        self.loss_function = None
        self.metric_class = None
        self.training_dataset_transformer = None
        self.validation_dataset_transformer = None
        self.training_minibatch_transformer = None
        self.validation_minibatch_transformer = None
        self.training_minibatch_preprocessor = None
        self.validation_minibatch_preprocessor = None
        self.minibatch_preprocessor = None
        self.coordinator_start_hook = None
        self.coordinator_stop_hook = None
        self.start_workers_hook = None
        self.stop_workers_hook = None
        self.worker_start_hook = None
        self.worker_stop_hook = None
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
        self.input_label_column_name = None
        self.output_label_column_name = None
        self.output_label_column_type = None
        self.output_prediction_column_name = None
        self.output_prediction_column_type = None
        self.minibatch_id = 0

    def run(self):
        if self.coordinator_start_hook is not None:
            self.coordinator_start_hook(self)
        self.distribute_module()
        self.distribute_updater()
        self.distribute_loss_function()
        self.distribute_metric_class()
        self.distribute_training_minibatch_transformer()
        self.distribute_validation_minibatch_transformer()
        self.distribute_training_minibatch_preprocessor()
        self.distribute_validation_minibatch_preprocessor()
        self.distribute_minibatch_preprocessor()
        self.distribute_worker_start_hook()
        self.distribute_worker_stop_hook()
        self.start_workers()
        self.feed_dataset()
        self.collect_module()
        self.stop_workers()
        if self.coordinator_stop_hook is not None:
            self.coordinator_stop_hook(self)

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

    def distribute_loss_function(self):
        loss_function = self.loss_function
        rdd = self.spark_context.parallelize(range(self.worker_count), self.worker_count)
        rdd.barrier().mapPartitions(lambda _: __class__._distribute_loss_function(loss_function, _)).collect()

    @classmethod
    def _distribute_loss_function(cls, loss_function, _):
        self = __class__.get_instance()
        self.loss_function = loss_function
        return _

    def distribute_metric_class(self):
        metric_class = self.metric_class
        rdd = self.spark_context.parallelize(range(self.worker_count), self.worker_count)
        rdd.barrier().mapPartitions(lambda _: __class__._distribute_metric_class(metric_class, _)).collect()

    @classmethod
    def _distribute_metric_class(cls, metric_class, _):
        self = __class__.get_instance()
        self.metric_class = metric_class
        return _

    def distribute_training_minibatch_transformer(self):
        transformer = self.training_minibatch_transformer
        rdd = self.spark_context.parallelize(range(self.worker_count), self.worker_count)
        rdd.barrier().mapPartitions(lambda _: __class__._distribute_training_minibatch_transformer(transformer, _)).collect()

    @classmethod
    def _distribute_training_minibatch_transformer(cls, transformer, _):
        self = __class__.get_instance()
        self.training_minibatch_transformer = transformer
        return _

    def distribute_validation_minibatch_transformer(self):
        transformer = self.validation_minibatch_transformer
        rdd = self.spark_context.parallelize(range(self.worker_count), self.worker_count)
        rdd.barrier().mapPartitions(lambda _: __class__._distribute_validation_minibatch_transformer(transformer, _)).collect()

    @classmethod
    def _distribute_validation_minibatch_transformer(cls, transformer, _):
        self = __class__.get_instance()
        self.validation_minibatch_transformer = transformer
        return _

    def distribute_training_minibatch_preprocessor(self):
        preprocessor = self.training_minibatch_preprocessor
        rdd = self.spark_context.parallelize(range(self.worker_count), self.worker_count)
        rdd.barrier().mapPartitions(lambda _: __class__._distribute_training_minibatch_preprocessor(preprocessor, _)).collect()

    @classmethod
    def _distribute_training_minibatch_preprocessor(cls, preprocessor, _):
        self = __class__.get_instance()
        self.training_minibatch_preprocessor = preprocessor
        return _

    def distribute_validation_minibatch_preprocessor(self):
        preprocessor = self.validation_minibatch_preprocessor
        rdd = self.spark_context.parallelize(range(self.worker_count), self.worker_count)
        rdd.barrier().mapPartitions(lambda _: __class__._distribute_validation_minibatch_preprocessor(preprocessor, _)).collect()

    @classmethod
    def _distribute_validation_minibatch_preprocessor(cls, preprocessor, _):
        self = __class__.get_instance()
        self.validation_minibatch_preprocessor = preprocessor
        return _

    def distribute_minibatch_preprocessor(self):
        preprocessor = self.minibatch_preprocessor
        rdd = self.spark_context.parallelize(range(self.worker_count), self.worker_count)
        rdd.barrier().mapPartitions(lambda _: __class__._distribute_minibatch_preprocessor(preprocessor, _)).collect()

    @classmethod
    def _distribute_minibatch_preprocessor(cls, preprocessor, _):
        self = __class__.get_instance()
        self.minibatch_preprocessor = preprocessor
        return _

    def distribute_worker_start_hook(self):
        worker_start_hook = self.worker_start_hook
        rdd = self.spark_context.parallelize(range(self.worker_count), self.worker_count)
        rdd.barrier().mapPartitions(lambda _: __class__._distribute_worker_start_hook(worker_start_hook, _)).collect()

    @classmethod
    def _distribute_worker_start_hook(cls, worker_start_hook, _):
        self = __class__.get_instance()
        self.worker_start_hook = worker_start_hook
        return _

    def distribute_worker_stop_hook(self):
        worker_stop_hook = self.worker_stop_hook
        rdd = self.spark_context.parallelize(range(self.worker_count), self.worker_count)
        rdd.barrier().mapPartitions(lambda _: __class__._distribute_worker_stop_hook(worker_stop_hook, _)).collect()

    @classmethod
    def _distribute_worker_stop_hook(cls, worker_stop_hook, _):
        self = __class__.get_instance()
        self.worker_stop_hook = worker_stop_hook
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

    def start_workers(self):
        if self.start_workers_hook is not None:
            self.start_workers_hook(self)
        super().start_workers()

    def worker_start(self):
        super().worker_start()
        self.setup_model()
        self.setup_trainer()
        self.load_model()
        if self.worker_start_hook is not None:
            self.worker_start_hook(self)

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

    def stop_workers(self):
        super().stop_workers()
        if self.stop_workers_hook is not None:
            self.stop_workers_hook(self)

    def worker_stop(self):
        # Make sure the final metric buffers are pushed.
        self.push_metric()
        if self.is_training_mode:
            self.save_model()
            self.export_model()
        if self.worker_stop_hook is not None:
            self.worker_stop_hook(self)
        super().worker_stop()

    def feed_dataset(self):
        if self.is_training_mode:
            self.feed_training_dataset()
        else:
            self.feed_validation_dataset()

    def feed_training_dataset(self):
        if self.training_dataset_transformer is not None:
            self.training_dataset_transformer(self)
        else:
            # For backward compatibility.
            self._default_feed_training_dataset()

    def _default_feed_training_dataset(self):
        from .input import shuffle_df
        for epoch in range(self.training_epoches):
            df = self.dataset
            if self.shuffle_training_dataset:
                df = shuffle_df(df, self.worker_count)
            func = self.feed_training_minibatch()
            df = df.mapInPandas(func, df.schema)
            df.write.format('noop').mode('overwrite').save()

    def feed_validation_dataset(self):
        if self.validation_dataset_transformer is not None:
            self.validation_dataset_transformer(self)
        else:
            # For backward compatibility.
            self._default_feed_validation_dataset()

    def _default_feed_validation_dataset(self):
        df = self.dataset
        func = self.feed_validation_minibatch()
        output_schema = self._make_validation_result_schema(df)
        df = df.mapInPandas(func, output_schema)
        self.validation_result = df
        # PySpark DataFrame & RDD is lazily evaluated.
        # We must call ``cache`` here otherwise PySpark will try to reevaluate
        # ``validation_result`` when we use it, which is not possible as the
        # PS system has been shutdown.
        df.cache()
        df.write.format('noop').mode('overwrite').save()

    def preprocess_minibatch(self, minibatch):
        if self.is_training_mode and self.training_minibatch_preprocessor is not None:
            result = self.training_minibatch_preprocessor(self, minibatch)
            return result
        elif not self.is_training_mode and self.validation_minibatch_preprocessor is not None:
            result = self.validation_minibatch_preprocessor(self, minibatch)
            return result
        elif self.minibatch_preprocessor is not None:
            result = self.minibatch_preprocessor(self, minibatch)
            return result
        else:
            # For backward compatibility.
            result = self._default_preprocess_minibatch(minibatch)
            return result

    def _default_preprocess_minibatch(self, minibatch):
        import numpy as np
        if self.input_label_column_name is not None:
            label_column_name = self.input_label_column_name
        else:
            label_column_name = minibatch.columns[self.input_label_column_index]
        labels = minibatch[label_column_name].values.astype(np.float32)
        return minibatch, labels

    def train_minibatch(self, minibatch):
        if self.training_minibatch_transformer is not None:
            self.training_minibatch_transformer(self, minibatch)
        else:
            # For backward compatibility.
            self._default_train_minibatch(minibatch)
        return minibatch

    def _default_train_minibatch(self, minibatch):
        self.model.train()
        minibatch, labels = self.preprocess_minibatch(minibatch)
        predictions = self.model(minibatch)
        labels = torch.from_numpy(labels).reshape(-1, 1)
        loss = self.compute_loss(predictions, labels)
        self.trainer.train(loss)
        self.update_progress(batch_size=len(minibatch), batch_loss=loss,
                             predictions=predictions, labels=labels)

    def validate_minibatch(self, minibatch):
        if self.validation_minibatch_transformer is not None:
            result = self.validation_minibatch_transformer(self, minibatch)
            return result
        else:
            # For backward compatibility.
            result = self._default_validate_minibatch(minibatch)
            return result

    def _default_validate_minibatch(self, minibatch):
        self.model.eval()
        minibatch, labels = self.preprocess_minibatch(minibatch)
        predictions = self.model(minibatch)
        labels = torch.from_numpy(labels).reshape(-1, 1)
        loss = self.compute_loss(predictions, labels)
        self.update_progress(batch_size=len(minibatch), batch_loss=loss,
                             predictions=predictions, labels=labels)
        return self._make_validation_result(minibatch, labels, predictions)

    def _make_validation_result(self, minibatch, labels, predictions):
        labels = labels.reshape(-1).numpy().astype(self.output_label_column_type)
        predictions = predictions.detach().reshape(-1).numpy().astype(self.output_prediction_column_type)
        minibatch[self.output_label_column_name] = labels
        minibatch[self.output_prediction_column_name] = predictions
        return minibatch

    def _make_validation_result_schema(self, df):
        from pyspark.sql.types import StructType
        fields = []
        reserved = set([self.output_label_column_name, self.output_prediction_column_name])
        for field in df.schema.fields:
            if field.name not in reserved:
                fields.append(field)
        result_schema = StructType(fields)
        result_schema.add(self.output_label_column_name, self.output_label_column_type)
        result_schema.add(self.output_prediction_column_name, self.output_prediction_column_type)
        return result_schema

    def compute_loss(self, predictions, labels):
        if self.loss_function is not None:
            loss = self.loss_function(predictions, labels)
            return loss
        else:
            # For backward compatibility.
            loss = self._default_compute_loss(predictions, labels)
            return loss

    def _default_compute_loss(self, predictions, labels):
        from .loss_utils import log_loss
        loss = log_loss(predictions, labels) / labels.shape[0]
        return loss

    def update_progress(self, **kwargs):
        self.minibatch_id += 1
        self.update_metric(**kwargs)
        if self.minibatch_id % self.metric_update_interval == 0:
            self.push_metric()

    def _get_metric_class(self):
        metric_class = self.metric_class
        if metric_class is not None:
            return metric_class
        return super()._get_metric_class()

class PyTorchLauncher(PSLauncher):
    def __init__(self):
        super().__init__()
        self.module = None
        self.updater = None
        self.dataset = None
        self.loss_function = None
        self.metric_class = None
        self.training_dataset_transformer = None
        self.validation_dataset_transformer = None
        self.training_minibatch_transformer = None
        self.validation_minibatch_transformer = None
        self.training_minibatch_preprocessor = None
        self.validation_minibatch_preprocessor = None
        self.minibatch_preprocessor = None
        self.coordinator_start_hook = None
        self.coordinator_stop_hook = None
        self.start_workers_hook = None
        self.stop_workers_hook = None
        self.worker_start_hook = None
        self.worker_stop_hook = None
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
        self.input_label_column_name = None
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
        agent.loss_function = self.loss_function
        agent.metric_class = self.metric_class
        agent.training_dataset_transformer = self.training_dataset_transformer
        agent.validation_dataset_transformer = self.validation_dataset_transformer
        agent.training_minibatch_transformer = self.training_minibatch_transformer
        agent.validation_minibatch_transformer = self.validation_minibatch_transformer
        agent.training_minibatch_preprocessor = self.training_minibatch_preprocessor
        agent.validation_minibatch_preprocessor = self.validation_minibatch_preprocessor
        agent.minibatch_preprocessor = self.minibatch_preprocessor
        agent.coordinator_start_hook = self.coordinator_start_hook
        agent.coordinator_stop_hook = self.coordinator_stop_hook
        agent.start_workers_hook = self.start_workers_hook
        agent.stop_workers_hook = self.stop_workers_hook
        agent.worker_start_hook = self.worker_start_hook
        agent.worker_stop_hook = self.worker_stop_hook
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
        self._agent_attributes['input_label_column_name'] = self.input_label_column_name
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
                 loss_function=None,
                 metric_class=None,
                 training_dataset_transformer=None,
                 validation_dataset_transformer=None,
                 training_minibatch_transformer=None,
                 validation_minibatch_transformer=None,
                 training_minibatch_preprocessor=None,
                 validation_minibatch_preprocessor=None,
                 minibatch_preprocessor=None,
                 coordinator_start_hook=None,
                 coordinator_stop_hook=None,
                 start_workers_hook=None,
                 stop_workers_hook=None,
                 worker_start_hook=None,
                 worker_stop_hook=None,
                 worker_count=1,
                 server_count=1,
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
                 input_label_column_index=None,
                 input_label_column_name='label',
                 output_label_column_name='label',
                 output_label_column_type='double',
                 output_prediction_column_name='rawPrediction',
                 output_prediction_column_type='double',
                 **kwargs):
        super().__init__()
        self.module = module
        self.updater = updater
        self.loss_function = loss_function
        self.metric_class = metric_class
        self.training_dataset_transformer = training_dataset_transformer
        self.validation_dataset_transformer = validation_dataset_transformer
        self.training_minibatch_transformer = training_minibatch_transformer
        self.validation_minibatch_transformer = validation_minibatch_transformer
        self.training_minibatch_preprocessor = training_minibatch_preprocessor
        self.validation_minibatch_preprocessor = validation_minibatch_preprocessor
        self.minibatch_preprocessor = minibatch_preprocessor
        self.coordinator_start_hook = coordinator_start_hook
        self.coordinator_stop_hook = coordinator_stop_hook
        self.start_workers_hook = start_workers_hook
        self.stop_workers_hook = stop_workers_hook
        self.worker_start_hook = worker_start_hook
        self.worker_stop_hook = worker_stop_hook
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
        self.input_label_column_name = input_label_column_name
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
        if self.loss_function is not None and not callable(self.loss_function):
            raise TypeError(f"loss_function must be callable; {self.loss_function!r} is invalid")
        if self.metric_class is not None and not issubclass(self.metric_class, ModelMetric):
            raise TypeError(f"metric_class must be a subclass of ModelMetric; {self.metric_class!r} is invalid")
        if self.training_dataset_transformer is not None and not callable(self.training_dataset_transformer):
            raise TypeError(f"training_dataset_transformer must be a subclass of ModelMetric; {self.training_dataset_transformer!r} is invalid")
        if self.validation_dataset_transformer is not None and not callable(self.validation_dataset_transformer):
            raise TypeError(f"validation_dataset_transformer must be a subclass of ModelMetric; {self.validation_dataset_transformer!r} is invalid")
        if self.training_minibatch_transformer is not None and not callable(self.training_minibatch_transformer):
            raise TypeError(f"training_minibatch_transformer must be a subclass of ModelMetric; {self.training_minibatch_transformer!r} is invalid")
        if self.validation_minibatch_transformer is not None and not callable(self.validation_minibatch_transformer):
            raise TypeError(f"validation_minibatch_transformer must be a subclass of ModelMetric; {self.validation_minibatch_transformer!r} is invalid")
        if self.training_minibatch_preprocessor is not None and not callable(self.training_minibatch_preprocessor):
            raise TypeError(f"training_minibatch_preprocessor must be a subclass of ModelMetric; {self.training_minibatch_preprocessor!r} is invalid")
        if self.validation_minibatch_preprocessor is not None and not callable(self.validation_minibatch_preprocessor):
            raise TypeError(f"validation_minibatch_preprocessor must be a subclass of ModelMetric; {self.validation_minibatch_preprocessor!r} is invalid")
        if self.minibatch_preprocessor is not None and not callable(self.minibatch_preprocessor):
            raise TypeError(f"minibatch_preprocessor must be a subclass of ModelMetric; {self.minibatch_preprocessor!r} is invalid")
        if self.coordinator_start_hook is not None and not callable(self.coordinator_start_hook):
            raise TypeError(f"coordinator_start_hook must be callable; {self.coordinator_start_hook!r} is invalid")
        if self.coordinator_stop_hook is not None and not callable(self.coordinator_stop_hook):
            raise TypeError(f"coordinator_stop_hook must be callable; {self.coordinator_stop_hook!r} is invalid")
        if self.start_workers_hook is not None and not callable(self.start_workers_hook):
            raise TypeError(f"start_workers_hook must be callable; {self.start_workers_hook!r} is invalid")
        if self.stop_workers_hook is not None and not callable(self.stop_workers_hook):
            raise TypeError(f"stop_workers_hook must be callable; {self.stop_workers_hook!r} is invalid")
        if self.worker_start_hook is not None and not callable(self.worker_start_hook):
            raise TypeError(f"worker_start_hook must be callable; {self.worker_start_hook!r} is invalid")
        if self.worker_stop_hook is not None and not callable(self.worker_stop_hook):
            raise TypeError(f"worker_stop_hook must be callable; {self.worker_stop_hook!r} is invalid")
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
        if self.input_label_column_index is not None:
            if not isinstance(self.input_label_column_index, int) or self.input_label_column_index < 0:
                raise TypeError(f"input_label_column_index must be non-negative integer; {self.input_label_column_index!r} is invalid")
        if self.input_label_column_name is not None and not isinstance(self.input_label_column_name, str):
            raise TypeError(f"input_label_column_name must be string; {self.input_label_column_name!r} is invalid")
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
        launcher.loss_function = self.loss_function
        launcher.metric_class = self.metric_class
        launcher.training_dataset_transformer = self.training_dataset_transformer
        launcher.validation_dataset_transformer = self.validation_dataset_transformer
        launcher.training_minibatch_transformer = self.training_minibatch_transformer
        launcher.validation_minibatch_transformer = self.validation_minibatch_transformer
        launcher.training_minibatch_preprocessor = self.training_minibatch_preprocessor
        launcher.validation_minibatch_preprocessor = self.validation_minibatch_preprocessor
        launcher.minibatch_preprocessor = self.minibatch_preprocessor
        launcher.coordinator_start_hook = self.coordinator_start_hook
        launcher.coordinator_stop_hook = self.coordinator_stop_hook
        launcher.start_workers_hook = self.start_workers_hook
        launcher.stop_workers_hook = self.stop_workers_hook
        launcher.worker_start_hook = self.worker_start_hook
        launcher.worker_stop_hook = self.worker_stop_hook
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
        launcher.input_label_column_name = self.input_label_column_name
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
        args['loss_function'] = self.loss_function
        args['metric_class'] = self.metric_class
        args['training_dataset_transformer'] = self.training_dataset_transformer
        args['validation_dataset_transformer'] = self.validation_dataset_transformer
        args['training_minibatch_transformer'] = self.training_minibatch_transformer
        args['validation_minibatch_transformer'] = self.validation_minibatch_transformer
        args['training_minibatch_preprocessor'] = self.training_minibatch_preprocessor
        args['validation_minibatch_preprocessor'] = self.validation_minibatch_preprocessor
        args['minibatch_preprocessor'] = self.minibatch_preprocessor
        args['coordinator_start_hook'] = self.coordinator_start_hook
        args['coordinator_stop_hook'] = self.coordinator_stop_hook
        args['start_workers_hook'] = self.start_workers_hook
        args['stop_workers_hook'] = self.stop_workers_hook
        args['worker_start_hook'] = self.worker_start_hook
        args['worker_stop_hook'] = self.worker_stop_hook
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
        args['input_label_column_name'] = self.input_label_column_name
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
        if self.model_out_path is not None:
            delete_dir(self.model_out_path)
        if self.model_export_path is not None:
            delete_dir(self.model_export_path)

    def _fit(self, dataset):
        self._clear_output()
        launcher = self._create_launcher(dataset, True)
        launcher.launch()
        module = launcher.agent_object.module
        module.eval()
        model = self._create_model(module)
        self.final_metric = launcher.agent_object._metric
        return model
