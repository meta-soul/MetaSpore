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

class PSLauncher(object):
    def __init__(self):
        self._agent_class = None
        self._worker_count = None
        self._server_count = None
        self._job_name = None
        self._keep_session = None
        self._spark_log_level = None
        self._agent_attributes = None

    def parse_args(self, args=None):
        import argparse
        parser = argparse.ArgumentParser(description="launcher for PS on PySpark")
        parser.add_argument('-a', '--agent-class', type=str, required=True,
            help="PS agent class to use")
        parser.add_argument('-w', '--worker-count', type=int, required=True,
            help="PS worker count")
        parser.add_argument('-s', '--server-count', type=int, required=True,
            help="PS server count")
        parser.add_argument('-j', '--job-name', type=str, required=True,
            help="Spark job name")
        parser.add_argument('-k', '--keep-session', action='store_true',
            help="Keep the Spark session from being shutdown")
        SPARK_LOG_LEVELS = 'ALL', 'DEBUG', 'ERROR', 'FATAL', 'INFO', 'OFF', 'TRACE', 'WARN'
        parser.add_argument('-L', '--spark-log-level', type=str, default='WARN',
            choices=SPARK_LOG_LEVELS, help="set Spark log level; default to WARN")
        parser.add_argument('--conf', type=str, action='append',
            help="pass NAME=VALUE option to PS agent as attributes")
        args = parser.parse_args(args)
        self._agent_class = args.agent_class
        self._worker_count = self._get_node_count(args, 'worker')
        self._server_count = self._get_node_count(args, 'server')
        self._job_name = args.job_name
        self._keep_session = args.keep_session
        self._spark_log_level = args.spark_log_level
        self._agent_attributes = self._get_agent_attributes(args)

    def _get_node_count(self, args, role):
        key = role + '_count'
        value = getattr(args, key)
        if value <= 0:
            message = "%s count must be positive; " % role
            message += "%d specified in command line is invalid" % value
            raise ValueError(message)
        return value

    def _get_agent_attributes(self, args):
        attrs = dict()
        if args.conf is not None:
            for item in args.conf:
                name, sep, value = item.partition('=')
                if not sep:
                    message = "'=' not found in --conf %s" % item
                    raise ValueError(message)
                attrs[name] = self._unnormalize_option_value(value)
        return attrs

    def _unnormalize_option_value(self, value):
        if value == 'null':
            return None
        if value == 'true':
            return True
        if value == 'false':
            return False
        try:
            result = int(value)
        except ValueError:
            pass
        else:
            return result
        try:
            result = float(value)
        except ValueError:
            pass
        else:
            return result
        return value

    def _split_agent_class_name(self):
        i = self._agent_class.rfind('.')
        if i == -1:
            message = "invalid agent class name: %s" % self._agent_class
            raise RuntimeError(message)
        module_name = self._agent_class[:i]
        class_name = self._agent_class[i+1:]
        return module_name, class_name

    def _get_agent_class(self):
        import os
        import sys
        import metaspore
        import importlib
        sys.path.insert(0, os.getcwd())
        module_name, class_name = self._split_agent_class_name()
        module = importlib.import_module(module_name)
        class_ = getattr(module, class_name, None)
        if class_ is None:
            message = "class %r not found in module %r" % (class_name, module_name)
            raise RuntimeError(message)
        if not isinstance(class_, type) or not issubclass(class_, metaspore.Agent):
            message = "%r is not a PS agent class" % self._agent_class
            raise RuntimeError(message)
        return class_

    def _initialize_agent(self, agent):
        pass

    def launch_agent(self):
        import asyncio
        import pyspark
        # Use nest_asyncio to workaround the problem that ``asyncio.run()``
        # can not be called in Jupyter Notebook toplevel.
        import nest_asyncio
        nest_asyncio.apply()
        class_ = self._get_agent_class()
        builder = pyspark.sql.SparkSession.builder
        if self._job_name is not None:
            builder.appName(self._job_name)
        spark_session = builder.getOrCreate()
        try:
            if self._spark_log_level is not None:
                spark_context = spark_session.sparkContext
                spark_context.setLogLevel(self._spark_log_level)
            args = dict()
            args['worker_count'] = self._worker_count
            args['server_count'] = self._server_count
            args['agent_attributes'] = self._agent_attributes
            asyncio.run(class_._launch(args, spark_session, self))
        finally:
            if not self._keep_session:
                spark_session.stop()

    def launch(self, args=None):
        self.parse_args(args)
        self.launch_agent()

def main():
    launcher = PSLauncher()
    launcher.launch()

if __name__ == '__main__':
    main()
