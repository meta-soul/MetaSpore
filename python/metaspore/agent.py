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
import threading
import concurrent.futures
import traceback
import json
import os
from ._metaspore import ActorConfig
from ._metaspore import NodeRole
from ._metaspore import Message
from ._metaspore import PSRunner
from ._metaspore import PSDefaultAgent
from .metric import BinaryClassificationModelMetric
from .network_utils import get_available_endpoint
from .url_utils import use_s3a

class Agent(object):
    _instances = dict()
    _instances_lock = threading.Lock()

    @property
    def _cxx_agent(self):
        cxx_agent = getattr(self, '_Agent__cxx_agent', None)
        if cxx_agent is None:
            message = "the adaptor agent is not initialized"
            raise RuntimeError(message)
        return cxx_agent

    def _finalize(self):
        # The C++ and Python agent objects reference each other.
        # The reference cycle are broken in the finalize methods.
        #
        # This complicated mechanism is needed to avoid deriving
        # the ``metaspore.Agent`` class from ``_metaspore.PSDefaultAgent``,
        # which makes it possible to define custom agent classes
        # in Jupyter notebooks.
        self.__cxx_agent = None

    def run(self):
        pass

    def handle_request(self, req):
        res = Message()
        self.send_response(req, res)

    @property
    def is_coordinator(self):
        return self._cxx_agent.is_coordinator

    @property
    def is_server(self):
        return self._cxx_agent.is_server

    @property
    def is_worker(self):
        return self._cxx_agent.is_worker

    @property
    def server_count(self):
        return self._cxx_agent.server_count

    @property
    def worker_count(self):
        return self._cxx_agent.worker_count

    @property
    def rank(self):
        return self._cxx_agent.rank

    def barrier(self, group=None):
        if group is None:
            self._cxx_agent.barrier()
        else:
            self._cxx_agent.barrier(group)

    def shutdown(self):
        self._cxx_agent.shutdown()

    def send_request(self, req, cb):
        self._cxx_agent.send_request(req, cb)

    def send_all_requests(self, reqs, cb):
        self._cxx_agent.send_all_requests(reqs, cb)

    def broadcast_request(self, req, cb):
        self._cxx_agent.broadcast_request(req, cb)

    def send_response(self, req, res):
        self._cxx_agent.send_response(req, res)

    def __str__(self):
        return self._cxx_agent.__str__()

    @property
    def spark_session(self):
        if not self.is_coordinator:
            message = "spark session is only available on coordinator"
            raise RuntimeError(message)
        session = getattr(self, '_Agent__spark_session', None)
        if session is None:
            message = "spark session is not initialized"
            raise RuntimeError(message)
        return session

    @property
    def spark_context(self):
        if not self.is_coordinator:
            message = "spark context is only available on coordinator"
            raise RuntimeError(message)
        session = getattr(self, '_Agent__spark_session', None)
        if session is None:
            message = "spark session is not initialized"
            raise RuntimeError(message)
        context = session.sparkContext
        return context

    @classmethod
    def _register_instance(cls, ident, instance):
        with cls._instances_lock:
            if ident in cls._instances:
                message = "more than one ps agents are registered for thread 0x%x" % ident
                raise RuntimeError(message)
            cls._instances[ident] = instance
            print('\033[38;5;046mps agent registered for process %d thread 0x%x\033[m' % (os.getpid(), ident))

    @classmethod
    def _deregister_instance(cls, ident):
        with cls._instances_lock:
            try:
                del cls._instances[ident]
                print('\033[38;5;196mps agent deregistered for process %d thread 0x%x\033[m' % (os.getpid(), ident))
            except KeyError:
                message = "during deregister instance, no ps agent registered for thread 0x%x on pid %d" % (threading.current_thread().ident, os.getpid())
                raise RuntimeError(message)

    @classmethod
    def get_instance(cls, ident=None):
        if ident is None:
            ident = threading.current_thread().ident
        with cls._instances_lock:
            instance = cls._instances.get(ident)
            if instance is None:
                message = "no ps agent registered for thread 0x%x on pid %d" % (ident, os.getpid())
                raise RuntimeError(message)
            return instance

    @classmethod
    def _get_actor_config(cls, args):
        conf = ActorConfig()
        conf.root_uri = args['root_uri']
        conf.root_port = args['root_port']
        conf.node_role = NodeRole.__members__[args['node_role']]
        conf.agent_creator = args.get('agent_creator', cls)
        agent_ready_callback = args.get('agent_ready_callback')
        if agent_ready_callback is not None:
            conf.agent_ready_callback = agent_ready_callback
        conf.server_count = args['server_count']
        conf.worker_count = args['worker_count']
        conf.is_message_dumping_enabled = args.get('is_message_dumping_enabled', False)
        return conf

    @classmethod
    def _get_reserved_attributes(cls):
        reserved = frozenset(dir(cls))
        return reserved

    @classmethod
    def _load_agent_attributes(cls, inst, args):
        attrs = args['agent_attributes']
        reserved = cls._get_reserved_attributes()
        for name, value in attrs.items():
            if name in reserved:
                message = "agent attribute %r is reserved, " % name
                message += "specifying it in config file and "
                message += "overriding it with --conf are forbidden"
                raise RuntimeError(message)
            setattr(inst, name, value)

    @classmethod
    def _create_agent(cls):
        cxx_agent = PSDefaultAgent()
        py_agent = cls()
        py_agent.__cxx_agent = cxx_agent
        cxx_agent.py_agent = py_agent
        return cxx_agent

    @classmethod
    def _run_server(cls, args, _):
        # Server processes block the spark method call so that later computational
        # spark method calls won't be performed on those server processes.
        def create_server():
            inst = cls._create_agent()
            cls._load_agent_attributes(inst.py_agent, args)
            return inst
        def server_ready(agent):
            print('PS Server node \033[38;5;196m%s\033[m is ready.' % agent)
        args = args.copy()
        args.update(agent_creator=create_server)
        args.update(agent_ready_callback=server_ready)
        args.update(node_role='Server')
        conf = cls._get_actor_config(args)
        PSRunner.run_ps(conf)
        return _

    @classmethod
    def _run_worker(cls, args, _):
        ident = threading.current_thread().ident
        ready = concurrent.futures.Future()
        def create_worker():
            inst = cls._create_agent()
            cls._load_agent_attributes(inst.py_agent, args)
            cls._register_instance(ident, inst.py_agent)
            return inst
        def worker_ready(agent):
            print('PS Worker node \033[38;5;051m%s\033[m is ready.' % agent)
            ready.set_result(None)
        args = args.copy()
        args.update(agent_creator=create_worker)
        args.update(agent_ready_callback=worker_ready)
        args.update(node_role='Worker')
        def run_worker():
            try:
                conf = cls._get_actor_config(args)
                PSRunner.run_ps(conf)
            except Exception:
                traceback.print_exc()
                raise SystemExit(1)
            finally:
                cls._deregister_instance(ident)
        # Worker processes must run in background mode so that the spark method call
        # can return immediately and later computational spark method calls can be
        # performed on those worker processes.
        thread = threading.Thread(target=run_worker, name='run_worker', daemon=True)
        thread.start()
        # Wait until the agent is ready which means all the PS nodes are connected to
        # each other.
        ready.result()
        return _

    @classmethod
    def _launch_coordinator(cls, args, spark_session, launcher):
        def create_coordinator():
            inst = cls._create_agent()
            inst.py_agent.__spark_session = spark_session
            cls._load_agent_attributes(inst.py_agent, args)
            launcher._initialize_agent(inst.py_agent)
            return inst
        def coordinator_ready(agent):
            print('PS Coordinator node \033[32m%s\033[m is ready.' % agent)
        args = args.copy()
        args.update(agent_creator=create_coordinator)
        args.update(agent_ready_callback=coordinator_ready)
        args.update(node_role='Coordinator')
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        def launch_coordinator():
            try:
                conf = cls._get_actor_config(args)
                PSRunner.run_ps(conf)
                loop.call_soon_threadsafe(future.set_result, None)
            except Exception as e:
                loop.call_soon_threadsafe(future.set_exception, e)
        thread = threading.Thread(target=launch_coordinator, name='launch_coordinator', daemon=True)
        thread.start()
        return future

    @classmethod
    def _launch_servers(cls, args, spark_session):
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        def launch_servers():
            try:
                server_count = args['server_count']
                spark_context = spark_session.sparkContext
                rdd = spark_context.parallelize(range(server_count), server_count)
                rdd.barrier().mapPartitions(lambda _: cls._run_server(args, _)).collect()
                loop.call_soon_threadsafe(future.set_result, None)
            except Exception as e:
                loop.call_soon_threadsafe(future.set_exception, e)
        thread = threading.Thread(target=launch_servers, name='launch_servers', daemon=True)
        thread.start()
        return future

    @classmethod
    def _launch_workers(cls, args, spark_session):
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        def launch_workers():
            try:
                worker_count = args['worker_count']
                spark_context = spark_session.sparkContext
                rdd = spark_context.parallelize(range(worker_count), worker_count)
                rdd.barrier().mapPartitions(lambda _: cls._run_worker(args, _)).collect()
                loop.call_soon_threadsafe(future.set_result, None)
            except Exception as e:
                loop.call_soon_threadsafe(future.set_exception, e)
        thread = threading.Thread(target=launch_workers, name='launch_workers', daemon=True)
        thread.start()
        return future

    @classmethod
    async def _launch(cls, args, spark_session, launcher):
        ip, port = get_available_endpoint()
        args = args.copy()
        args.update(root_uri=ip)
        args.update(root_port=port)
        futures = []
        futures.append(cls._launch_servers(args, spark_session))
        futures.append(cls._launch_workers(args, spark_session))
        futures.append(cls._launch_coordinator(args, spark_session, launcher))
        await asyncio.gather(*futures)

    def worker_start(self):
        pass

    def worker_stop(self):
        pass

    @classmethod
    def _worker_start(cls, _):
        self = __class__.get_instance()
        self.worker_start()
        return _

    @classmethod
    def _worker_stop(cls, _):
        self = __class__.get_instance()
        self.worker_stop()
        return _

    def start_workers(self):
        rdd = self.spark_context.parallelize(range(self.worker_count), self.worker_count)
        rdd.barrier().mapPartitions(self._worker_start).collect()

    def stop_workers(self):
        rdd = self.spark_context.parallelize(range(self.worker_count), self.worker_count)
        rdd.barrier().mapPartitions(self._worker_stop).collect()

    def load_dataset(self, dataset_path):
        from pyspark.sql import functions as F
        dataset_path = use_s3a(dataset_path)
        df = (self.spark_session.read
              .format('csv')
              .option('header', 'false')
              .option('nullable', 'false')
              .option('delimiter', '\002')
              .option('encoding', 'UTF-8')
              .load(dataset_path))
        return df.select(F.array(df.columns))

    def feed_training_dataset(self, dataset_path, nepoches=1):
        for epoch in range(nepoches):
            df = self.load_dataset(dataset_path)
            func = self.feed_training_minibatch()
            df = df.mapInPandas(func, df.schema)
            df.write.format('noop').mode('overwrite').save()

    def feed_validation_dataset(self, dataset_path, nepoches=1):
        for epoch in range(nepoches):
            df = self.load_dataset(dataset_path)
            func = self.feed_validation_minibatch()
            df = df.mapInPandas(func, df.schema)
            df.write.format('noop').mode('overwrite').save()

    def feed_training_minibatch(self):
        def _feed_training_minibatch(iterator):
            self = __class__.get_instance()
            for minibatch in iterator:
                result = self.train_minibatch(minibatch)
                yield  result
        return _feed_training_minibatch

    def feed_validation_minibatch(self):
        def _feed_validation_minibatch(iterator):
            self = __class__.get_instance()
            for minibatch in iterator:
                result = self.validate_minibatch(minibatch)
                yield result
        return _feed_validation_minibatch

    def train_minibatch(self, minibatch):
        message = "Agent.train_minibatch method is not implemented"
        raise NotImplementedError(message)

    def validate_minibatch(self, minibatch):
        message = "Agent.validate_minibatch method is not implemented"
        raise NotImplementedError(message)

    def _get_metric_class(self):
        return BinaryClassificationModelMetric

    def _create_metric(self):
        metric_class = self._get_metric_class()
        metric = metric_class()
        return metric

    @property
    def _metric(self):
        metric = getattr(self, '_Agent__metric', None)
        if metric is None:
            metric = self._create_metric()
            self.__metric = metric
        return metric

    def update_metric(self, **kwargs):
        self._metric.accumulate(**kwargs)

    def push_metric(self):
        body = dict(command='PushMetric')
        req = Message()
        req.body = json.dumps(body)
        req.receiver = 0 << 4 | 8 | 1
        states = self._metric.get_states()
        for state in states:
            req.add_slice(state)
        def push_metric_callback(req, res):
            self.clear_metric()
        self.send_request(req, push_metric_callback)

    def clear_metric(self):
        self._metric.clear()

    def handle_request(self, req):
        body = json.loads(req.body)
        command = body.get('command')
        if command == 'PushMetric':
            states = ()
            for i in range(req.slice_count):
                states += req.get_slice(i),
            accum = self._metric
            delta = self._create_metric()
            delta.from_states(states)
            accum.merge(delta)
            string = accum.format(delta)
            print(string)
            res = Message()
            self.send_response(req, res)
            return
        super().handle_request(req)
