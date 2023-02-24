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

class ModelArtsEntrypointWrapper(object):
    def __init__(self):
        import threading
        self._barrier_thread = None
        self._barrier_process = None
        self._master_thread = None
        self._master_process = None
        self._worker_thread = None
        self._worker_process = None
        self._entrypoint_argv = None

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback):
        if self._worker_process is not None:
            self._worker_process.kill()
            self._worker_process = None
        if self._master_process is not None:
            self._master_process.kill()
            self._master_process = None
        if self._barrier_process is not None:
            self._barrier_process.kill()
            self._barrier_process = None
        if self._worker_thread is not None:
            self._worker_thread.join()
            self._worker_thread = None
        if self._master_thread is not None:
            self._master_thread.join()
            self._master_thread = None
        if self._barrier_thread is not None:
            self._barrier_thread.join()
            self._barrier_thread = None

    def _log_message(self, message):
        import sys
        print(message, file=sys.stdout)
        print(message, file=sys.stderr)
        sys.stdout.flush()
        sys.stderr.flush()

    @property
    def _master_port(self):
        return 7077

    @property
    def _webui_port(self):
        return 8080

    @property
    def _barrier_port(self):
        return 6060

    @property
    def _worker_work_dir(self):
        import os
        import tempfile
        path = os.path.join(tempfile.gettempdir(), 'spark_worker')
        return path

    @property
    def _is_master(self):
        import os
        host_name = os.environ['HOSTNAME']
        prefix, _, rank = host_name.rpartition('-')
        return rank == '0'

    @property
    def _spark_master(self):
        import os
        host_name = os.environ['HOSTNAME']
        domain_name = os.environ['MA_VJ_NAME']
        prefix, _, rank = host_name.rpartition('-')
        host = prefix + '-0.' + domain_name
        port = self._master_port
        url = 'spark://%s:%d' % (host, port)
        return url

    @property
    def _spark_webui(self):
        url = self._spark_master
        url = url.replace('spark://', 'http://')
        url = url.replace(':%d' % self._master_port, ':%d' % self._webui_port)
        return url

    @property
    def _spark_barrier(self):
        url = self._spark_master
        url = url.replace('spark://', 'http://')
        url = url.replace(':%d' % self._master_port, ':%d' % self._barrier_port)
        return url

    @property
    def _driver_host(self):
        url = self._spark_master
        url = url.replace('spark://', '')
        url = url.replace(':%d' % self._master_port, '')
        return url

    @property
    def _alive_workers(self):
        import re
        import requests
        try:
            res = requests.get(self._spark_webui)
        except OSError:
            return 0
        text = res.text
        match = re.search('<li><strong>Alive Workers:</strong> (\d+)</li>', text)
        if match is None:
            return 0
        else:
            return int(match.group(1))

    @property
    def _total_workers(self):
        import os
        num = int(os.environ['MA_NUM_HOSTS'])
        return num

    def _parse_args(self):
        import argparse
        parser = argparse.ArgumentParser(description='ModelArts entrypoint wrapper')
        parser.add_argument('command', type=str,
            help='wrapped entrypoint command')
        parser.add_argument('args', type=str, nargs='*', metavar='arg',
            help='arguments for wrapped entrypoint')
        args = parser.parse_args()
        self._entrypoint_argv = tuple([args.command] + args.args)

    def _spark_barrier_thread(self):
        import sys
        import subprocess
        args = sys.executable, '-m'
        args += 'uvicorn', 'metaspore.deploy.utils.spark_barrier:app'
        args += '--host', '0.0.0.0'
        args += '--port', str(self._barrier_port)
        with subprocess.Popen(args) as process:
            self._barrier_process = process
            process.wait()

    def _spark_master_thread(self):
        import subprocess
        args = 'spark-class', 'org.apache.spark.deploy.master.Master'
        with subprocess.Popen(args) as process:
            self._master_process = process
            process.wait()

    def _spark_worker_thread(self):
        import subprocess
        args = 'spark-class', 'org.apache.spark.deploy.worker.Worker'
        args += '--work-dir', self._worker_work_dir
        args += self._spark_master,
        with subprocess.Popen(args) as process:
            self._worker_process = process
            process.wait()

    def _start_barrier(self):
        import threading
        thread = threading.Thread(target=self._spark_barrier_thread, name='spark_barrier', daemon=True)
        thread.start()
        self._barrier_thread = thread

    def _start_master(self):
        import threading
        thread = threading.Thread(target=self._spark_master_thread, name='spark_master', daemon=True)
        thread.start()
        self._master_thread = thread

    def _start_worker(self):
        import requests
        import threading
        thread = threading.Thread(target=self._spark_worker_thread, name='spark_worker', daemon=True)
        thread.start()
        self._worker_thread = thread

    def _notify_barrier(self):
        import requests
        requests.get(self._spark_barrier + '/notify')

    def _wait_barrier(self):
        import time
        import requests
        count = 60 * 5
        while count > 0:
            count -= 1
            try:
                requests.get(self._spark_barrier + '/wait')
                self._log_message('Wait Spark barrier succeed.')
                return
            except OSError:
                pass
            time.sleep(1)
        self._log_message('Fail to wait Spark barrier.')
        raise SystemExit

    def _wait_cluster(self):
        import time
        count = 60 * 5
        while count > 0:
            count -= 1
            workers = self._alive_workers
            if workers == self._total_workers:
                self._log_message('Spark standalone cluster is ready.')
                return
            time.sleep(1)
        self._log_message('Fail to start Spark standalone cluster.')
        raise SystemExit

    def _wait_cleanup(self):
        import time
        time.sleep(5)

    def _config_env(self):
        import os
        os.environ['METASPORE_SPARK_MASTER'] = self._spark_master
        os.environ['METASPORE_SPARK_DRIVER_HOST'] = self._driver_host

    def _submit_job(self):
        import subprocess
        subprocess.check_call(self._entrypoint_argv)

    def run(self):
        with self:
            self._parse_args()
            self._log_message('Spark Master: %s' % self._spark_master)
            if self._is_master:
                self._start_barrier()
                self._start_master()
                self._wait_cluster()
                self._config_env()
                self._submit_job()
                self._notify_barrier()
                self._wait_cleanup()
            else:
                self._start_worker()
                self._wait_barrier()

def main():
    entry = ModelArtsEntrypointWrapper()
    entry.run()

if __name__ == '__main__':
    main()
