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
import sys
import importlib
import inspect
import argparse
import asyncio
import grpc
import metaspore_pb2
import metaspore_pb2_grpc

class PreprocessorServicer(metaspore_pb2_grpc.PredictServicer):
    def __init__(self, preprocessor_object):
        super().__init__()
        self._preprocessor_object = preprocessor_object

    async def Predict(self,
                      request: metaspore_pb2.PredictRequest,
                      context: grpc.aio.ServicerContext) -> metaspore_pb2.PredictReply:
        inputs = request.payload
        if inspect.iscoroutinefunction(self._preprocessor_object.predict):
            outputs = await self._preprocessor_object.predict(inputs)
        else:
            outputs = self._preprocessor_object.predict(inputs)
        reply = metaspore_pb2.PredictReply(payload=outputs)
        return reply

class PreprocessorService(object):
    def __init__(self):
        pass

    def _parse_args(self):
        parser = argparse.ArgumentParser(description="MetaSpre Serving preprocessor service")
        parser.add_argument('-c', '--config-dir', type=str, required=True,
            help="config directory of the processor; containing the 'processor.py' script and "
                 "its support files")
        parser.add_argument('-a', '--listen-addr', type=str, required=True,
            help="gRPC listen address of the preprocessor service")
        args = parser.parse_args()
        self._config_dir = args.config_dir
        self._listen_addr = args.listen_addr

    def _get_preprocessor_class(self):
        sys.path.insert(0, self._config_dir)
        module = importlib.import_module('preprocessor')
        preprocessor_classes = []
        for name in dir(module):
            if name.endswith('Preprocessor'):
                item = getattr(module, name)
                if isinstance(item, type):
                    preprocessor_classes.append(item)
        py_path = os.path.join(self._config_dir, 'preprocessor.py')
        if not preprocessor_classes:
            message = "no preprocessor class found in %r" % py_path
            raise RuntimeError(message)
        if len(preprocessor_classes) == 1:
            return preprocessor_classes[0]
        exported = getattr(module, '__all__', None)
        if exported is not None:
            candicates = [c for c in preprocessor_classes if c.__name__ in exported]
            if len(candicates) == 1:
                return candicates[0]
        message = "more than one preprocessor classes found in %r" % py_path
        raise RuntimeError(message)

    def _load_preprocessor(self):
        preprocessor_class = self._get_preprocessor_class()
        preprocessor = preprocessor_class()
        preprocessor.load(self._config_dir)
        self._preprocessor_object = preprocessor

    def _print_input_output_names(self):
        input_names = ','.join(self._preprocessor_object.input_names)
        output_names = ','.join(self._preprocessor_object.output_names)
        print('input_names=%s' % input_names)
        print('output_names=%s' % output_names)
        sys.stdout.flush()

    async def _serve(self):
        servicer = PreprocessorServicer(self._preprocessor_object)
        server = grpc.aio.server()
        metaspore_pb2_grpc.add_PredictServicer_to_server(servicer, server)
        server.add_insecure_port(self._listen_addr)
        await server.start()
        self._print_input_output_names()
        await server.wait_for_termination()

    def run(self):
        self._parse_args()
        self._load_preprocessor()
        asyncio.run(self._serve())

def main():
    service = PreprocessorService()
    service.run()

if __name__ == '__main__':
    main()
