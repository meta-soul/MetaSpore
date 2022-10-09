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
import random
import time

import consul


class Consul(object):
    def __init__(self, host, port, token=None):
        self._consul = consul.Consul(host, port, token=token)

    def setConfig(self, key, value):
        try:
            return self._consul.kv.put(key, value, None)
        except Exception as e:
            import traceback
            traceback.print_exc()
            return False

    def getConfig(self, key):
        index, data = self._consul.kv.get(key)
        return data['Value']

    def deletConfig(self, key):
        self._consul.kv.delete(key)


def putServiceConfig(config, host="localhost", port=8500, prefix="config", context="recommend", data_key="data"):
    client = Consul(host, port)
    key = "%s/%s/%s" % (prefix, context, data_key)
    num = 14
    while num > 0 and not client.setConfig(key, config):
        print("wait set config to consul!")
        time.sleep(1)
        num -= 1
    print("set config to consul success!")
