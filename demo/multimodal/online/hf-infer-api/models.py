import os
import json
import base64
import requests


class TextCompletionModel:

    def __init__(self, model_name=None, model_type=None):
        self.api_token = os.environ.get("HF_API_TOKEN", "")
        self.model_name = "gpt2" if not model_name else model_name

    def get_params(self, args={}):
        return {
            "do_sample": True,
            "top_k": args.get("top_k", 30),
            "top_p": args.get("top_p", 1.0),
            "return_full_text": False,
            "max_new_tokens": args.get("max_length",0) if args.get("max_length",0) else args.get("max_new_tokens", 128),
            "num_return_sequences": args.get("num_sentences", 0) if args.get("num_sentences", 0) else args.get("num_return_sequences", 1)
        }

    def pre_process(self, req, pending=False):
        return {
            "inputs": req.inputs,
            "parameters": self.get_params(json.loads(req.args)),
            "options": {
                "use_cache": True,
                "wait_for_model": pending
            }
        }

    def post_process(self, res):
        ret = {"data": []}
        for x in json.loads(res.content.decode("utf-8")):
            ret["data"].append(x["generated_text"])
        return ret

    def __call__(self, req, api_token=""):
        api_token = api_token if api_token else self.api_token
        assert api_token, "API token is empty!"

        headers = {"Authorization": f"Bearer {api_token}"}
        api_url = f"https://api-inference.huggingface.co/models/{self.model_name}"
        try:
            payload = self.pre_process(req)
            response = requests.request("POST", api_url, 
                headers=headers, data=json.dumps(payload))
            # retry
            if response.status_code == 503:
                print(f"[RETRY] [model={self.model_name}]")
                payload = self.pre_process(req, pending=True)
                response = requests.request("POST", api_url,
                    headers=headers, data=json.dumps(payload))
            return self.post_process(response)
        except Exception as e:
            print(f"[ERROR] {e}")
            return {"data": None}


class Text2TextModel:

    def __init__(self, model_name=None, model_type=None):
        self.api_token = os.environ.get("HF_API_TOKEN", "")
        self.model_name = "google/mt5-base" if not model_name else model_name

    def get_params(self, args={}):
        return {
            "do_sample": True,
            "top_k": args.get("top_k", 30),
            "top_p": args.get("top_p", 1.0),
            "max_new_tokens": args.get("max_length", 0) if args.get("max_length", 0) else args.get("max_new_tokens",128),
            "num_return_sequences": args.get("num_sentences", 0) if args.get("num_sentences", 0) else args.get("num_return_sequences", 1)
        }

    def pre_process(self, req, pending=False):
        return {
            "inputs": req.inputs,
            "parameters": self.get_params(json.loads(req.args)),
            "options": {
                "use_cache": True,
                "wait_for_model": pending
            }
        }

    def post_process(self, res):
        ret = {"data": []}
        for x in json.loads(res.content.decode("utf-8")):
            ret["data"].append(x["generated_text"])
        return ret

    def __call__(self, req, api_token=""):
        api_token = api_token if api_token else self.api_token
        assert api_token, "API token is empty!"

        headers = {"Authorization": f"Bearer {api_token}"}
        api_url = f"https://api-inference.huggingface.co/models/{self.model_name}"
        try:
            payload = self.pre_process(req)
            response = requests.request("POST", api_url, 
                headers=headers, data=json.dumps(payload))
            # retry
            if response.status_code == 503:
                print(f"[RETRY] [model={self.model_name}]")
                payload = self.pre_process(req, pending=True)
                response = requests.request("POST", api_url,
                    headers=headers, data=json.dumps(payload))
            return self.post_process(response)
        except Exception as e:
            print(f"[ERROR] {e}")
            return {"data": None}

class TranslationModel:

    def __init__(self, model_name=None, model_type="zh2en"):
        self.api_token = os.environ.get("HF_API_TOKEN", "")
        self.model_name = model_name
        if not self.model_name:
            if model_type == "zh2en":
                self.model_name = "Helsinki-NLP/opus-mt-zh-en"
            else:
                self.model_name = "Helsinki-NLP/opus-mt-en-zh"

    def get_params(self, args={}):
        return {
            "do_sample": True,
            "top_k": args.get("top_k", 30),
            "top_p": args.get("top_p", 1.0),
            "num_return_sequences": args.get("num_sentences", 0) if args.get("num_sentences", 0) else args.get("num_return_sequences", 1)
        }

    def pre_process(self, req, pending=False):
        return {
            "inputs": req.inputs,
            #"parameters": self.get_params(json.loads(req.args)),
            "options": {
                "use_cache": True,
                "wait_for_model": pending
            }
        }

    def post_process(self, res):
        ret = {"data": []}
        for x in json.loads(res.content.decode("utf-8")):
            ret["data"].append(x["translation_text"])
        return ret

    def __call__(self, req, api_token=""):
        api_token = api_token if api_token else self.api_token
        assert api_token, "API token is empty!"

        headers = {"Authorization": f"Bearer {api_token}"}
        api_url = f"https://api-inference.huggingface.co/models/{self.model_name}"
        try:
            payload = self.pre_process(req)
            response = requests.request("POST", api_url, 
                headers=headers, data=json.dumps(payload))
            # retry
            if response.status_code == 503:
                print(f"[RETRY] [model={self.model_name}]")
                payload = self.pre_process(req, pending=True)
                response = requests.request("POST", api_url,
                    headers=headers, data=json.dumps(payload))
            return self.post_process(response)
        except Exception as e:
            print(f"[ERROR] {e}")
            return {"data": None}


class Text2ImageModel:

    def __init__(self, model_name=None, model_type=None):
        self.api_token = os.environ.get("HF_API_TOKEN", "")
        self.model_name = "IDEA-CCNL/Taiyi-Stable-Diffusion-1B-Chinese-v0.1" if not model_name else model_name

    def get_params(self, args={}):
        return {}

    def pre_process(self, req, pending=False):
        #return req.inputs
        return {
            "inputs": req.inputs,
            "options": {
                "use_cache": True,
                "wait_for_model": pending
            }
        }

    def post_process(self, res):
        ret = {"data": base64.b64encode(res.content).decode('utf8')}
        return ret

    def __call__(self, req, api_token=""):
        api_token = api_token if api_token else self.api_token
        assert api_token, "API token is empty!"

        headers = {"Authorization": f"Bearer {api_token}"}
        api_url = f"https://api-inference.huggingface.co/models/{self.model_name}"
        try:
            payload = self.pre_process(req)
            response = requests.request("POST", api_url, 
                headers=headers, data=json.dumps(payload))
            # retry
            if response.status_code == 503:
                print(f"[RETRY] [model={self.model_name}]")
                payload = self.pre_process(req, True)
                response = requests.request("POST", api_url,
                    headers=headers, data=json.dumps(payload))
            return self.post_process(response)
        except Exception as e:
            print(f"[ERROR] {e}")
            return {"data": None}
