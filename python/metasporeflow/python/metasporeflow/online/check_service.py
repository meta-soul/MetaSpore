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
import json

import requests
import time
import traceback

def notifyRecommendService(host, port):
    print("notify recommend service %s:%s" % (host, port))
    max_wait = 300
    num = max_wait
    last_exception = None
    while num > 0:
        try:
            resp = requests.post('http://%s:%s/actuator/refresh' % (host, port))
        except Exception as ex:
            resp = None
            last_exception = ex
        if resp is not None and resp.status_code == 200:
            try:
                data = resp.json()
            except Exception as ex:
                data = None
                last_exception = ex
            if data is not None:
                # succeed: print and return
                print(data)
                return
        print("retry refresh recommend service! %s:%s" % (host, port))
        time.sleep(1)
        num -= 1
    if last_exception is not None:
        traceback.print_exception(last_exception)
    message = "fail to notify recommend service %s:%s after waiting %d seconds" % (host, port, max_wait)
    raise RuntimeError(message)


def healthRecommendService(host, port):
    try:
        resp = requests.get('http://%s:%s/actuator/health' % (host, port))
    except Exception as ex:
        return {"status": "DOWN", "resp": None, "msg": "health check request fail, ex:{}".format(ex.args)}
    if resp is not None:
        if resp.status_code != 200:
            return {"status": "DOWN", "resp": resp,
                    "msg": "health check request fail, ret_code:{}".format(resp.status_code)}
        else:
            try:
                data = resp.json()
            except Exception as ex:
                return {"status": "DOWN", "resp": resp,
                        "msg": "health check request resp parser fail, ex:{}".format(ex.args)}
            if data is not None:
                status = data.setdefault("status", "DOWN")
                if status == "OUT_OF_SERVICE":
                    return {"status": "DOWN", "resp": data, "msg": "recommend has empty config"}
                elif status == "DOWN":
                    return {"status": "DOWN", "resp": data, "msg": "health check fail!"}
                return {"status": status, "resp": data, "msg": "health check successfully"}
    return {"status": "DOWN", "resp": None, "msg": "health check request fail, unknown"}


def tryRecommendService(host, port, scene, param=None):
    if param is None:
        param = {"user_id": "test"}
    try:
        header = {
            'Content-Type': 'application/json'
        }
        resp = requests.post('http://%s:%s/service/recommend/%s' % (host, port, scene), headers=header, data=json.dumps(param))
    except Exception as ex:
        return {"status": "DOWN", "resp": None, "msg": "request scene:{} fail, ex:{}".format(scene, ex.args)}
    if resp is not None:
        if resp.status_code != 200:
            return {"status": "DOWN", "resp": resp,
                    "msg": "request scene:{}, ret_code:{}".format(scene, resp.status_code)}
        else:
            try:
                data = resp.json()
            except Exception as ex:
                return {"status": "DOWN", "resp": resp,
                        "msg": "request scene:{} request resp parser fail, ex:{}".format(scene, ex.args)}
            if data is not None:
                data["timeRecords"] = None
                code = data.setdefault("code", "UNKNOWN")
                if code != "SUCCESS":
                    return {"status": "DOWN", "resp": data, "msg": "recommend request scene: {} fail".format(scene)}
                result = data.get("data", [])
                if len(result) == 0:
                    return {"status": 'DOWN', "resp": data,
                            "msg": "recommend request scene: {} return empty result".format(scene)}
                data["data"] = None
                return {"status": 'UP', "resp": data, "msg": "recommend request scene: {} successfully".format(scene)}
    return {"status": "DOWN", "resp": None, "msg": "recommend request scene: {} fail, unknown".format(scene)}


if __name__ == "__main__":
    print("test")
    data = tryRecommendService("recommend-k8s-service-saas-demo.huawei.dmetasoul.com", 80, "guess-you-like")
    print(data)
