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
import asyncio
import grpc
import metaspore_pb2
import metaspore_pb2_grpc

async def main():
    server_addr = 'unix://' + os.getcwd() + '/listen_addr.sock'
    async with grpc.aio.insecure_channel(server_addr) as channel:
        stub = metaspore_pb2_grpc.PredictStub(channel)
        request = metaspore_pb2.PredictRequest()
        request.payload['input_key1'] = b'abc'
        request.payload['input_key2'] = b'def'
        print('request:')
        print(request)
        print()
        reply = await stub.Predict(request)
        print('reply:')
        print(reply)
        assert reply.payload['output_key1'] == b'def'
        assert reply.payload['output_key2'] == b'abc'

if __name__ == '__main__':
    asyncio.run(main())
