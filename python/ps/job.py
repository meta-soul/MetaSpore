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

import argparse
import glob
import os
import re
import sys
import subprocess
import yaml

class JobRunnder(object):
    def __init__(self):
        self._debug_mode = None
        self._agent_class = None
        self._user_name = None
        self._spark_log_level = None
        self._job_config = None
        self._cmdline_args = None
        self._is_local_mode = None
        self._batch_size = None
        self._worker_count = None
        self._server_count = None
        self._worker_cpu = None
        self._server_cpu = None
        self._worker_memory = None
        self._server_memory = None
        self._python_env = None
        self._python_ver = None
        self._archives = None
        self._py_files = None
        self._files = None
        self._jars = []

    def parse_args(self):
        parser = argparse.ArgumentParser(description="job runner for PS on PySpark")
        parser.add_argument('-d', '--debug-mode', action='store_true',
            help="log commands for debugging purpose")
        parser.add_argument('-a', '--agent-class', type=str,
            help="PS agent class to use")
        parser.add_argument('-u', '--user-name', type=str, required=True,
            help="PySpark job user name; required for cluster job")
        parser.add_argument('-j', '--job-name', type=str,
            help="PySpark job name; generate one if not specified")
        parser.add_argument('-c', '--job-config', type=str,
            help="job config YAML file path; relative to -C if specified")
        parser.add_argument('-b', '--batch-size', type=int,
            help="override batch size specified in config file")
        parser.add_argument('-w', '--worker-count', type=int,
            help="override worker count specified in config file")
        parser.add_argument('-s', '--server-count', type=int,
            help="override server count specified in config file")
        parser.add_argument('--worker-cpu', type=int,
            help="override worker cpu specified in config file")
        parser.add_argument('--server-cpu', type=int,
            help="override server cpu specified in config file")
        parser.add_argument('--worker-memory', type=str,
            help="override worker memory specified in config file")
        parser.add_argument('--server-memory', type=str,
            help="override server memory specified in config file")
        parser.add_argument('-e', '--python-env', type=str,
            help="override python-env.tgz specified in config file")
        parser.add_argument('-v', '--python-ver', type=str,
            help="override Python version specified in config file; "
                 "default to the version of the current Python interpreter")
        parser.add_argument('--conf', type=str, action='append',
            help="pass NAME=VALUE option to PS agent as attributes")
        parser.add_argument('--spark-conf', type=str, action='append',
            help="pass extra NAME=VALUE --conf options to spark-submit")
        parser.add_argument('--spark-env', type=str, action='append',
            help="pass extra NAME=VALUE environment options to spark-submit")
        parser.add_argument('--spark-archives', type=str, action='append',
            help="pass extra --archives options to spark-submit")
        parser.add_argument('--spark-py-files', type=str, action='append',
            help="pass extra --py-files options to spark-submit")
        parser.add_argument('--spark-files', type=str, action='append',
            help="pass extra --files options to spark-submit")
        parser.add_argument('--spark-jars', type=str, action='append',
            help="pass extra --jars options to spark-submit")
        SPARK_LOG_LEVELS = 'ALL', 'DEBUG', 'ERROR', 'FATAL', 'INFO', 'OFF', 'TRACE', 'WARN'
        parser.add_argument('-L', '--spark-log-level', type=str, default='WARN',
            choices=SPARK_LOG_LEVELS, help="set Spark log level; default to WARN")
        parser.add_argument('-C', '--chdir', type=str,
            help="execute 'chdir' before run the job")
        parser.add_argument('--local', action='store_true',
            help="run local mode job")
        parser.add_argument('--cluster', action='store_true',
            help="run cluster mode job")
        parser.add_argument('extra_args', type=str, nargs='*',
            help="pass extra arguments to spark-submit, "
                 "only available when no agent class is specified; "
                 "the '--' argument separator may be used")
        args = parser.parse_args()
        if not args.local and not args.cluster:
            message = "one of --local and --cluster must be specified"
            raise RuntimeError(message)
        if args.local and args.cluster:
            message = "only one of --local and --cluster can be specified"
            raise RuntimeError(message)
        if not args.agent_class and not args.extra_args:
            message = "one of --agent-class and extra_args must be specified"
            raise RuntimeError(message)
        if args.agent_class and args.extra_args:
            message = "only one of --agent-class and extra_args can be specified"
            raise RuntimeError(message)
        if args.chdir:
            os.chdir(args.chdir)
        self._debug_mode = args.debug_mode
        self._agent_class = args.agent_class
        self._user_name = args.user_name
        self._job_name = args.job_name
        self._spark_log_level = args.spark_log_level
        if args.job_config is None:
            self._job_config = dict()
        else:
            with open(args.job_config) as fin:
                self._job_config = yaml.full_load(fin)
        self._cmdline_args = args
        self._is_local_mode = args.local
        conf = self._get_spark_config(args)
        self._batch_size = self._get_batch_size(args, conf)
        self._worker_count = self._get_node_count(args, conf, 'worker')
        self._server_count = self._get_node_count(args, conf, 'server')
        self._worker_cpu = self._get_node_cpu(args, conf, 'worker')
        self._server_cpu = self._get_node_cpu(args, conf, 'server')
        self._worker_memory = self._get_node_memory(args, conf, 'worker')
        self._server_memory = self._get_node_memory(args, conf, 'server')
        if not args.local:
            self._python_env = self._get_node_python_env(args, conf)
            self._python_ver = self._get_node_python_ver(args, conf)

    def _get_spark_config(self, args):
        if args.local:
            conf = self._job_config.get('local')
            if conf is None:
                conf = dict()
        else:
            conf = self._job_config.get('cluster')
            if conf is None:
                conf = self._job_config.get('distributed')
                if conf is None:
                    conf = dict()
        return conf

    def _get_batch_size(self, args, conf):
        key = 'batch_size'
        value = getattr(args, key)
        if value is not None:
            if value <= 0:
                message = "batch size must be positive integer; "
                message += "%d specified in command line is invalid" % value
                raise ValueError(message)
            return value
        value = conf.get(key)
        if value is None:
            message = "batch size is not specified in command line nor config file"
            raise RuntimeError(message)
        if not isinstance(value, int) or value <= 0:
            message = "batch size must be positive integer; "
            message += "%r specified in config file is invalid" % value
            raise ValueError(message)
        return value

    def _get_node_count(self, args, conf, role):
        key = role + '_count'
        value = getattr(args, key)
        if value is not None:
            if value <= 0:
                message = "%s count must be positive; " % role
                message += "%d specified in command line is invalid" % value
                raise ValueError(message)
            return value
        value = conf.get(key)
        if value is None:
            alt_key = role + 's'
            value = conf.get(alt_key)
            if value is None:
                message = "%s count is not specified in command line nor config file" % role
                raise RuntimeError(message)
        if not isinstance(value, int) or value <= 0:
            message = "%s count must be positive integer; " % role
            message += "%r specified in config file is invalid" % value
            raise ValueError(message)
        return value

    def _get_node_cpu(self, args, conf, role):
        key = role + '_cpu'
        value = getattr(args, key)
        if value is not None:
            if value <= 0:
                message = "%s cpu must be positive; " % role
                message += "%d specified in command line is invalid" % value
                raise ValueError(message)
            return value
        value = conf.get(key)
        if value is None:
            if self._is_local_mode:
                # This is not used in local mode, return a dummy value.
                return 1
            message = "%s cpu is not specified in command line nor config file" % role
            raise RuntimeError(message)
        if not isinstance(value, int) or value <= 0:
            message = "%s cpu must be positive integer; " % role
            message += "%r specified in config file is invalid" % value
            raise ValueError(message)
        return value

    def _get_node_memory(self, args, conf, role):
        key = role + '_memory'
        value = getattr(args, key)
        if value is not None:
            return value
        value = conf.get(key)
        if value is None:
            if self._is_local_mode:
                # This is not used in local mode, return a dummy value.
                return '1G'
            message = "%s memory is not specified in command line nor config file" % role
            raise RuntimeError(message)
        return value

    def _get_node_python_env(self, args, conf):
        key = 'python_env'
        value = getattr(args, key)
        if value is not None:
            return value
        value = conf.get(key)
        if value is None:
            key2 = 'python-env'
            value = conf.get(key2)
            if value is None:
                message = "python-env is not specified in command line nor config file"
                raise RuntimeError(message)
        return value

    def _get_node_python_ver(self, args, conf):
        key = 'python_ver'
        value = getattr(args, key)
        if value is not None:
            return value
        value = conf.get(key)
        if value is not None:
            return value
        v = sys.version_info
        value = '%s.%s.%s' % (v.major, v.minor, v.micro)
        return value

    def _check_python_env(self):
        if os.path.isfile(self._python_env):
            pass
        elif os.path.isdir(self._python_env):
            py_ver = '.'.join(self._python_ver.split('.')[:-1])
            ma_dir = 'lib/python%s/site-packages/metaspore' % py_ver
            ma_path = os.path.join(self._python_env, ma_dir)
            if not os.path.isdir(ma_path):
                message = "%r is not a valid python-env, " % self._python_env
                message += "because MetaSpore is not found in it"
                raise RuntimeError(message)
            tgz_path = self._python_env + '.tgz'
            tgz_mtime = 0.0
            if os.path.isfile(tgz_path):
                tgz_mtime = os.path.getmtime(tgz_path)
            dir_mtime = os.path.getmtime(self._python_env)
            if dir_mtime > tgz_mtime:
                args = ['tar', '-czf', tgz_path, '-C', self._python_env] + os.listdir(self._python_env)
                subprocess.check_call(args)
            self._python_env = tgz_path
        else:
            message = "python-env %r not found" % self._python_env
            raise RuntimeError(message)

    def _normalize_option_value(self, value):
        if isinstance(value, str):
            return value
        if value is None:
            return 'null'
        if value is True:
            return 'true'
        if value is False:
            return 'false'
        return str(value)

    def _get_driver_memory(self):
        return '5G'

    def _get_executor_memory(self):
        from metaspore import job_utils
        mem = job_utils.merge_storage_size(self._worker_memory, self._server_memory)
        return mem

    def _get_executor_count(self):
        num = self._worker_count + self._server_count
        return str(num)

    def _get_executor_cores(self):
        return str(self._worker_cpu)

    def _get_launcher_local_path(self):
        from metaspore import ps_launcher
        path = ps_launcher.__file__
        return path

    def _get_python_executable_path(self):
        if self._is_local_mode:
            python_path = sys.executable
        else:
            python_path = './python-env/bin/python'
        return python_path

    def _get_cluster_ld_library_path(self):
        ld_library_path = './python-env/lib'
        return ld_library_path

    def _get_spark_submit_command(self):
        python_path = self._get_python_executable_path()
        args = ['env']
        args += ['PYSPARK_PYTHON=%s' % python_path]
        args += ['PYSPARK_DRIVER_PYTHON=%s' % python_path]
        args += ['spark-submit']
        return args

    def _get_spark_master_config(self):
        if self._is_local_mode:
            args = ['--master', 'local[%s]' % self._get_executor_count()]
        else:
            args = ['--master', 'yarn', '--deploy-mode', 'cluster', '--name', self._get_job_name()]
        return args

    def _get_spark_executors_config(self):
        conf = dict()
        python_path = self._get_python_executable_path()
        conf['spark.sql.execution.arrow.maxRecordsPerBatch'] = self._batch_size
        conf['spark.yarn.appMasterEnv.PYSPARK_PYTHON'] = python_path
        conf['spark.yarn.appMasterEnv.PYSPARK_DRIVER_PYTHON'] = python_path
        conf['spark.executorEnv.PYSPARK_PYTHON'] = python_path
        conf['spark.executorEnv.PYSPARK_DRIVER_PYTHON'] = python_path
        if not self._is_local_mode:
            ld_library_path = self._get_cluster_ld_library_path()
            conf['spark.yarn.appMasterEnv.LD_LIBRARY_PATH'] = ld_library_path
            conf['spark.executorEnv.LD_LIBRARY_PATH'] = ld_library_path
            conf['spark.executorEnv.PYTHONPATH'] = ''
            conf['spark.executorEnv.PYTHONNOUSERSITE'] = '1'
        conf['spark.python.worker.reuse'] = 'true'
        conf['spark.dynamicAllocation.enabled'] = 'false'
        conf['spark.shuffle.service.enabled'] = 'false'
        conf['spark.sql.execution.arrow.pyspark.enabled'] = 'true'
        conf['spark.task.maxFailures'] = '1'
        conf['spark.yarn.maxAppAttempts'] = '1'
        conf['spark.scheduler.minRegisteredResourcesRatio'] = '1.0'
        conf['spark.scheduler.maxRegisteredResourcesWaitingTime'] = '1800s'
        spark_conf = self._job_config.get('spark_conf')
        spark_env = self._job_config.get('spark_env')
        if spark_conf is not None:
            conf.update(spark_conf)
        if spark_env is not None:
            for name, value in spark_env.items():
                if self._is_local_mode and name == 'PYTHONPATH':
                    continue
                conf['spark.yarn.appMasterEnv.%s' % name] = value
                conf['spark.executorEnv.%s' % name] = value
        if self._cmdline_args.spark_conf is not None:
            for item in self._cmdline_args.spark_conf:
                name, sep, value = item.partition('=')
                if not sep:
                    message = "'=' not found in --spark-conf %s" % item
                    raise ValueError(message)
                conf[name] = value
        if self._cmdline_args.spark_env is not None:
            for item in self._cmdline_args.spark_env:
                name, sep, value = item.partition('=')
                if not sep:
                    message = "'=' not found in --spark-env %s" % item
                    raise ValueError(message)
                if self._is_local_mode and name == 'PYTHONPATH':
                    continue
                conf['spark.yarn.appMasterEnv.%s' % name] = value
                conf['spark.executorEnv.%s' % name] = value
        args = []
        for name, value in conf.items():
            value = self._normalize_option_value(value)
            args += ['--conf', '%s=%s' % (name, value)]
        return args

    def _get_spark_resources_config(self):
        args = []
        if not self._is_local_mode:
            args += ['--driver-memory', self._get_driver_memory()]
            args += ['--num-executors', self._get_executor_count()]
            args += ['--executor-memory', self._get_executor_memory()]
            args += ['--executor-cores', self._get_executor_cores()]
            args += ['--conf', 'spark.task.cpus=%s' % self._get_executor_cores()]
            args += ['--conf', 'spark.kubernetes.executor.request.cores=%s' % self._get_executor_cores()]
            args += ['--conf', 'spark.executorEnv.OMP_NUM_THREADS=%s' % self._get_executor_cores()]
        return args

    def _get_spark_files_config(self):
        args = []
        from metaspore.url_utils import use_s3a
        if not self._is_local_mode:
            args += ['--archives', use_s3a(','.join(self._archives))]
            args += ['--py-files', use_s3a(','.join(self._py_files))]
            args += ['--files', use_s3a(','.join(self._files))]
        if self._jars is not None and len(self._jars) > 0:
            args += ['--jars', use_s3a(','.join(self._jars))]
        return args

    def _get_job_name(self):
        import re
        from datetime import datetime
        from metaspore import network_utils
        from metaspore import __version__
        if self._job_name is not None:
            return self._job_name
        timestamp = datetime.now().strftime('%Y_%m_%d__%H_%M_%S__%f')
        host_ip = network_utils.get_host_ip()
        host_ip = re.sub(r'\W', '_', host_ip)
        user_name = re.sub(r'\W', '_', self._user_name)
        if self._agent_class:
            class_name = re.sub(r'\W', '_', self._agent_class)
        else:
            class_name = 'NoAgentClass'
        ps_version = re.sub(r'\W', '_', __version__)
        job_name = f'ML__{timestamp}__{host_ip}__{user_name}__{class_name}__PS_{ps_version}'
        return job_name

    def _get_ps_launcher_config(self):
        if not self._agent_class:
            return self._cmdline_args.extra_args
        args = [self._get_launcher_local_path()]
        args += ['--agent-class', str(self._agent_class)]
        args += ['--worker-count', str(self._worker_count)]
        args += ['--server-count', str(self._server_count)]
        args += ['--job-name', self._get_job_name()]
        args += ['--spark-log-level', self._spark_log_level]
        conf = dict()
        agent_conf = self._job_config.get('agent')
        if agent_conf is not None:
            conf.update(agent_conf)
        if self._cmdline_args.conf is not None:
            for item in self._cmdline_args.conf:
                name, sep, value = item.partition('=')
                if not sep:
                    message = "'=' not found in --conf %s" % item
                    raise ValueError(message)
                conf[name] = value
        for name, value in conf.items():
            value = self._normalize_option_value(value)
            args += ['--conf', '%s=%s' % (name, value)]
        return args

    def find_files(self):
        # jars should be included for local mode
        config_jars = self._job_config.get('spark_jars')
        if config_jars is not None:
            self._jars.extend([config_jars] if type(config_jars) == str else config_jars)
        if self._cmdline_args.spark_jars is not None:
            jars = self._cmdline_args.spark_jars
            self._jars.extend([jars] if type(jars) == str else jars)

        # ignore archives, py_files, files for local mode
        if self._is_local_mode:
            return
        self._check_python_env()
        archives = []
        py_files = []
        files = []
        def scan_files(dir_path):
            items = os.listdir(dir_path)
            for item in items:
                item_path = os.path.join(dir_path, item)
                if os.path.isdir(item_path):
                    if item in ('python-env', '.pyenv', '.git') or item.startswith('spark-'):
                        continue
                    scan_files(item_path)
                elif os.path.isfile(item_path):
                    if item.endswith('.py'):
                        py_files.append(item_path + '#' + item_path)
                    elif item.endswith('.yaml'):
                        files.append(item_path + '#' + item_path)
                    elif item.endswith('.txt'):
                        if item.startswith('column_name') or item.startswith('combine_schema'):
                            files.append(item_path + '#' + item_path)
        archives.append(self._python_env + '#python-env')
        scan_files('.')
        spark_archives = self._job_config.get('spark_archives')
        spark_py_files = self._job_config.get('spark_py_files')
        spark_files = self._job_config.get('spark_files')
        if spark_archives is not None:
            for name, path in spark_archives.items():
                archives.append('%s#%s' % (path, name))
        if spark_py_files is not None:
            for name, path in spark_py_files.items():
                py_files.append('%s#%s' % (path, name))
        if spark_files is not None:
            for name, path in spark_files.items():
                files.append('%s#%s' % (path, name))
        if self._cmdline_args.spark_archives is not None:
            for item in self._cmdline_args.spark_archives:
                archives.append(item)
        if self._cmdline_args.spark_py_files is not None:
            for item in self._cmdline_args.spark_py_files:
                py_files.append(item)
        if self._cmdline_args.spark_files is not None:
            for item in self._cmdline_args.spark_files:
                files.append(item)
        self._archives = tuple(archives)
        self._py_files = tuple(py_files)
        self._files = tuple(files)

    def spark_submit(self):
        args = self._get_spark_submit_command()
        args += self._get_spark_master_config()
        args += self._get_spark_executors_config()
        args += self._get_spark_resources_config()
        args += self._get_spark_files_config()
        args += self._get_ps_launcher_config()
        if self._debug_mode:
            from metaspore import shell_utils
            shell_utils.log_command(args)
        try:
            subprocess.check_call(args)
        except subprocess.CalledProcessError as e:
            from metaspore import shell_utils
            message = "spark-submit command failed with exit code %d" % e.returncode
            shell_utils.log_command(args)
            shell_utils.log_error(message)
            raise RuntimeError(message) from e

    def run(self):
        self.parse_args()
        self.find_files()
        self.spark_submit()

def main():
    runner = JobRunnder()
    runner.run()

if __name__ == '__main__':
    main()
