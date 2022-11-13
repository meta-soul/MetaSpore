import json
import os
import requests


class ModelHandler(object):

    def __init__(self):
        self.initialized = False
        self.recommend_host = "127.0.0.1"
        self.recommend_port = 13013

    def notify_model_path(self, model_dir):
        model_path = os.path.join(model_dir, "models")
        if not os.path.isdir(model_path):
            raise RuntimeError("Missing model {} path.".format(model_path))
        for model_name in os.listdir(model_path):
            if not os.path.isdir(os.path.join(model_path, model_name)):
                print("Missing model {}.".format(model_name))
                continue
            model_info = dict()
            model_info["modelName"] = model_name
            model_info["version"] = "1"
            model_info["dirPath"] = os.path.join(model_path, model_name)
            model_info["host"] = "127.0.0.1"
            model_info["port"] = 50000
            self.notifyLoadModel(model_info, self.recommend_host, self.recommend_port)

    def notify_model_info(self, model_dir, model_info_prefix):
        model_info_path = os.path.join(model_dir, "{}-{}".format(model_info_prefix, "model-info.json"))
        if not os.path.isfile(model_info_path):
            raise RuntimeError("Missing model {} file.".format(model_info_path))
        with open(model_info_path) as f:
            self.model_infos = json.load(f)
        if isinstance(self.model_infos, list):
            for model_info in self.model_infos:
                self.notifyLoadModel(model_info, self.recommend_host, self.recommend_port)
        elif isinstance(self.model_infos, dict):
            self.notifyLoadModel(self.model_infos, self.recommend_host, self.recommend_port)
        else:
            raise RuntimeError("Format error {} file.".format(model_info_path))

    def set_service_config(self, model_dir, config_prefix):
        config_file_path = os.path.join(model_dir, "{}-{}".format(config_prefix, "service-config.yaml"))
        if not os.path.isfile(config_file_path):
            raise RuntimeError("Missing config {} file.".format(config_file_path))
        with open(config_file_path) as f:
            self.setRecommendServiceConfig(f.read(), self.recommend_host, self.recommend_port)

    def setRecommendServiceConfig(self, config, host="127.0.0.1", port=8080):
        params = dict()
        params["operator"] = "updateconfig"
        params["config"] = config
        try:
            header = {
                'Content-Type': 'application/json'
            }
            resp = requests.post('http://%s:%s/invocations' % (host, port), headers=header,
                                 data=json.dumps(params))
            print("notify set config resp:", resp)
        except Exception as ex:
            print("notify set config fail param:", params)

    def notifyLoadModel(self, params, host="127.0.0.1", port=13013):
        try:
            params["operator"] = "loadmodel"
            if "servingName" not in params:
                if "host" not in params:
                    params["host"] = "127.0.0.1"
                if "port" not in params:
                    params["port"] = 50000
            header = {
                'Content-Type': 'application/json'
            }
            resp = requests.post('http://%s:%s/invocations' % (host, port), headers=header,
                                 data=json.dumps(params))
            print("notify load model resp:", resp)
        except Exception as ex:
            print("notify load model fail param:", params)

    def initialize(self, context):
        properties = context.system_properties
        # Contains the url parameter passed to the load request
        model_dir = properties.get("model_dir")
        self.set_service_config(model_dir, "recommend")
        self.notify_model_path(model_dir)
        self.initialized = True

    def preprocess(self, request):
        input = []
        for idx, data in enumerate(request):
            body = data.get("body")
            info = json.load(body)
            if "operator" in info:
                input.append(info)
            else:
                print("request is not match")
        return input

    def requestService(self, params, host="127.0.0.1", port=13013):
        try:
            header = {
                'Content-Type': 'application/json'
            }
            resp = requests.post('http://%s:%s/invocations' % (host, port), headers=header,
                                 data=json.dumps(params))
            if resp is not None:
                if resp.status_code != 200:
                    print("request service fail retcode: %s param:" % resp.status_code, params)
                else:
                    try:
                        return resp.json()
                    except Exception as ex:
                        print("request service result parser fail param:", params)
        except Exception as ex:
            print("request service fail param:", params)

    def inference(self, model_input):
        res = list()
        for item in model_input:
            res.append(self.requestService(item, self.recommend_host, self.recommend_port))
        return res

    def postprocess(self, inference_output):
        return inference_output

    def handle(self, data, context):
        model_input = self.preprocess(data)
        model_out = self.inference(model_input)
        return self.postprocess(model_out)


_service = ModelHandler()


def handle(data, context):
    if not _service.initialized:
        _service.initialize(context)

    if data is None:
        return None

    return _service.handle(data, context)
