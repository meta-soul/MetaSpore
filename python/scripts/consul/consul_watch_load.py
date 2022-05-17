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
import asyncio
import base64
import collections
import dataclasses
import json
import logging
import os
import sys
import traceback

from dataclasses import dataclass
from typing import Optional

@dataclass(frozen=True)
class ModelInfo:
    name: str
    service: str
    path: str
    version: str
    util_cmd: str

@dataclass(frozen=True)
class ConsulKeyChange:
    CreateIndex: int
    Flags: int
    Key: str
    LockIndex: int
    ModifyIndex: int
    Session: str
    Value: Optional[str]

@dataclass(frozen=True)
class DecodedConsulKeyChange:
    CreateIndex: int
    Flags: int
    Key: str
    LockIndex: int
    ModifyIndex: int
    Session: str
    Value: ModelInfo

class ModelWatcher():
    def __init__(self):
        self._key_info_dict = {}
        self._key_lock_dict = collections.defaultdict(asyncio.Lock)

    def _parse_args(self):
        default_prefix = 'dev/'
        default_model_root = os.path.abspath('models') + os.path.sep
        default_notify_port = 50051
        parser = argparse.ArgumentParser(description='Watch consul keyprefx and notify loading models')
        parser.add_argument('--prefix', type=str,
                            help=f"prefix to watch; must end with '/'; default to {default_prefix!r}")
        parser.add_argument('--model-root', type=str,
                            help=f"model root dir; default to {default_model_root!r}")
        parser.add_argument('--notify-port', type=int,
                            help=f"model load notify port; default to {default_notify_port!r}")
        args, left = parser.parse_known_args()
        sys.argv = sys.argv[:1] + left

        self._prefix = args.prefix
        if self._prefix is None:
            self._prefix = default_prefix
            print(f'Using default prefix {self._prefix!r}', file=sys.stderr)
        if not self._prefix.endswith('/'):
            message = f"watch prefix must end with '/'; {self._prefix!r} is invalid"
            raise RuntimeError(message)

        self._model_root = args.model_root
        if self._model_root is None:
            self._model_root = default_model_root
            print(f'Using default model root {self._model_root!r}', file=sys.stderr)

        self._notify_port = args.notify_port
        if self._notify_port is None:
            self._notify_port = default_notify_port
            print(f'Using default notify port {self._notify_port!r}', file=sys.stderr)

        pod_ip = os.environ.get('POD_IP')
        if pod_ip is None:
            pod_ip = '127.0.0.1'
            print(f'Using default POD_IP {pod_ip!r}', file=sys.stderr)
        self._log_extra = { 'clientip': pod_ip }

    def _config_logger(self):
        ch = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s %(clientip)-15s %(message)s')
        ch.setFormatter(formatter)
        self._logger = logging.getLogger('watcher')
        self._logger.setLevel(logging.INFO)
        self._logger.addHandler(ch)
        self._logger.info(f'Start watching prefix {self._prefix!r} to load models', extra=self._log_extra)

    def _start_watch(self):
        from aiohttp import web
        app = web.Application()
        app.add_routes([web.post('/notify', self._watch_handler)])
        web.run_app(app)

    def run(self):
        self._parse_args()
        self._config_logger()
        self._start_watch()

    async def _watch_handler(self, request):
        from aiohttp import web
        data = await request.json()
        if data is not None:
            changes = self._decode_changes(data)
            if changes is not None:
                await self._process_changes(changes)
        return web.Response()

    def _decode_changes(self, data):
        import cattrs
        from typing import List
        try:
            changes = cattrs.structure(data, List[ConsulKeyChange])
            return changes
        except Exception:
            traceback.print_exc()
            string = json.dumps(data, separators=(',', ': '), indent=4)
            self._logger.error('Fail to decode keyprefix changes: %s', string, extra=self._log_extra)
            return None

    def _decode_model_info(self, data):
        import cattrs
        try:
            json_str = base64.b64decode(data)
            dict_obj = json.loads(json_str)
            info = cattrs.structure(dict_obj, ModelInfo)
            return info
        except Exception:
            traceback.print_exc()
            self._logger.error('Fail to decode model info: %s', data, extra=self._log_extra)
            return None

    async def _process_changes(self, keyprefix_changes):
        self._logger.info('Received keyprefix changes: %s', keyprefix_changes, extra=self._log_extra)
        for key_info in keyprefix_changes:
            if key_info.Key is None:
                continue
            if key_info.Key == self._prefix:
                continue
            if key_info.Value is None:
                continue
            if key_info.Key in self._key_info_dict:
                old_info = self._key_info_dict[key_info.Key]
                if old_info.ModifyIndex != key_info.ModifyIndex:
                    self._logger.info(
                        '%s modified, notify service to reload', key_info.Key, extra=self._log_extra)
                    loop = asyncio.get_event_loop()
                    loop.create_task(self._load_change(key_info))
                else:
                    self._logger.info(
                        '%s modify index unchanged, ignored', key_info.Key, extra=self._log_extra)
            else:
                self._logger.info(
                    'Key %s not found, notify service to load', key_info.Key, extra=self._log_extra)
                print(self._key_info_dict)
                loop = asyncio.get_event_loop()
                loop.create_task(self._load_change(key_info))

    def _get_model_local_path(self, model_info):
        local_path = os.path.join(self._model_root, model_info.name, model_info.version)
        local_path = os.path.realpath(local_path)
        if not local_path.endswith(os.path.sep):
            local_path += os.path.sep
        return local_path

    async def _download_model(self, model_info):
        self._logger.info('Downloading model %s', model_info, extra=self._log_extra)
        local_path = self._get_model_local_path(model_info)
        proc = await asyncio.create_subprocess_exec(
                'aws', 's3', 'sync', '--delete', model_info.path, local_path)
        retcode = await proc.wait()
        if retcode == 0:
            self._logger.info('Successfully downloaded model %s', model_info, extra=self._log_extra)
            return True
        else:
            self._logger.error('Fail to download model %s', model_info, extra=self._log_extra)
            return False

    async def _notify_loading_model(self, model_info):
        import grpc
        import metaspore_pb2
        import metaspore_pb2_grpc
        self._logger.info('Notify loading model %s', model_info, extra=self._log_extra)
        local_path = self._get_model_local_path(model_info)
        try:
            async with grpc.aio.insecure_channel('0.0.0.0:%d' % self._notify_port) as channel:
                stub = metaspore_pb2_grpc.LoadStub(channel)
                request = metaspore_pb2.LoadRequest(model_name=model_info.name, version=model_info.version, dir_path=local_path)
                reply = await stub.Load(request)
                self._logger.info('OK: %s', reply.msg, extra=self._log_extra)
                return True
        except Exception:
            traceback.print_exc()
            self._logger.error('Fail to notify loading model %s, local_path: %s', model_info, local_path, extra=self._log_extra)
            return False

    async def _load_change(self, key_info):
        model_info = self._decode_model_info(key_info.Value)
        if model_info is None:
            return
        self._logger.info('Loading model %s', model_info, extra=self._log_extra)
        kwargs = dataclasses.asdict(key_info)
        kwargs['Value'] = model_info
        key_info = DecodedConsulKeyChange(**kwargs)
        async with self._key_lock_dict[key_info.Key]:
            succ = await self._download_model(model_info)
            if not succ:
                return
            succ = await self._notify_loading_model(model_info)
            if not succ:
                return
            self._key_info_dict[key_info.Key] = key_info
            self._logger.info('Successfully loaded model %s', model_info, extra=self._log_extra)
            print(self._key_info_dict)

def main():
    watcher = ModelWatcher()
    watcher.run()

if __name__ == '__main__':
    main()
