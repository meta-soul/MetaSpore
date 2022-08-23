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

class SessionBuilder(object):
    def __init__(self,
                 local=False,
                 batch_size=100,
                 worker_count=1,
                 server_count=1,
                 worker_cpu=1,
                 server_cpu=1,
                 worker_memory='5G',
                 server_memory='5G',
                 coordinator_memory='5G',
                 app_name=None,
                 spark_master=None,
                 log_level='WARN',
                 spark_confs={}):
        self.local = local
        self.batch_size = batch_size
        self.worker_count = worker_count
        self.server_count = server_count
        self.worker_cpu = worker_cpu
        self.server_cpu = server_cpu
        self.worker_memory = worker_memory
        self.server_memory = server_memory
        self.coordinator_memory = coordinator_memory
        self.app_name = app_name
        self.spark_master = spark_master
        self.log_level = log_level
        self.spark_confs = spark_confs

    def _get_executor_count(self):
        num = self.worker_count + self.server_count
        return num

    def _config_app_name(self, builder):
        app_name = self.app_name
        if app_name is None:
            if self._is_interactive():
                app_name = 'MetaSpore-Notebook'
            else:
                app_name = 'MetaSpore-Job'
        builder.appName(app_name)

    def _is_interactive(self):
        try:
            ipython = get_ipython()
        except NameError:
            return False  # Probably standard Python interpreter
        name = ipython.__class__.__name__
        if name == 'ZMQInteractiveShell':
            return True   # Jupyter Notebook or QtConsole
        elif name == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)

    def _config_spark_master(self, builder):
        if self.local:
            master = 'local[%d]' % self._get_executor_count()
            builder.master(master)
        else:
            if self.spark_master is not None:
                builder.master(self.spark_master)

    def _config_batch_size(self, builder):
        builder.config('spark.sql.execution.arrow.maxRecordsPerBatch', str(self.batch_size))

    def _config_resources(self, builder):
        from . import job_utils
        builder.config('spark.driver.memory', self.coordinator_memory)
        executor_memory = job_utils.merge_storage_size(self.worker_memory, self.server_memory)
        builder.config('spark.executor.memory', executor_memory)
        builder.config('spark.executor.instances', str(self._get_executor_count()))
        num_threads = max(self.worker_cpu, self.server_cpu)
        builder.config('spark.executor.cores', str(num_threads))
        builder.config('spark.task.cpus', str(num_threads))
        builder.config('spark.kubernetes.executor.request.cores', str(num_threads))
        builder.config('spark.executorEnv.OMP_NUM_THREADS', str(num_threads))

    def _config_env(self, builder, name, value):
        import os
        if value:
            value = str(value)
            os.environ[name] = value
        else:
            value = ''
            os.environ.unsetenv(name)
        builder.config('spark.executorEnv.%s' % (name,), value)
        builder.config('spark.yarn.appMasterEnv.%s' % (name,), value)

    def _add_extra_configs(self, builder):
        builder.config('spark.python.worker.reuse', 'true')
        builder.config('spark.dynamicAllocation.enabled', 'false')
        builder.config('spark.shuffle.service.enabled', 'false')
        builder.config('spark.sql.execution.arrow.pyspark.enabled', 'true')
        builder.config('spark.task.maxFailures', '1')
        builder.config('spark.yarn.maxAppAttempts', '1')
        builder.config('spark.scheduler.minRegisteredResourcesRatio', '1.0')
        builder.config('spark.scheduler.maxRegisteredResourcesWaitingTime', '1800s')

    def _add_s3_configs(self, builder):
        from .s3_utils import get_s3_config
        config = get_s3_config()
        self._config_env(builder, 'AWS_REGION', config.aws_region)
        self._config_env(builder, 'AWS_ENDPOINT', config.aws_endpoint)
        self._config_env(builder, 'AWS_ACCESS_KEY_ID', config.aws_access_key_id)
        self._config_env(builder, 'AWS_SECRET_ACCESS_KEY', config.aws_secret_access_key)
        if config.aws_endpoint:
            builder.config('spark.hadoop.fs.s3a.endpoint', config.aws_endpoint)
        builder.config('spark.hadoop.fs.s3a.impl', 'org.apache.hadoop.fs.s3a.S3AFileSystem')
        builder.config('spark.hadoop.fs.s3.impl', 'org.apache.hadoop.fs.s3a.S3AFileSystem')
        builder.config('spark.hadoop.fs.oss.impl', 'org.apache.hadoop.fs.s3a.S3AFileSystem')

    def _add_user_spark_configs(self, builder):
        for k, v in self.spark_confs.items():
            builder.config(k, v)

    def build(self):
        import pyspark
        builder = pyspark.sql.SparkSession.builder
        self._config_app_name(builder)
        self._config_spark_master(builder)
        self._config_batch_size(builder)
        self._config_resources(builder)
        self._add_extra_configs(builder)
        self._add_s3_configs(builder)
        self._add_user_spark_configs(builder)
        spark_session = builder.getOrCreate()
        spark_context = spark_session.sparkContext
        spark_context.setLogLevel(self.log_level)
        return spark_session

def get_session(*args, **kwargs):
    builder = SessionBuilder(*args, **kwargs)
    spark_session = builder.build()
    return spark_session
