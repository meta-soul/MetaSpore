import json
import requests

res = requests.post('http://127.0.0.1:8098/api/infer/text-to-image', 
    data=json.dumps({"inputs": "连衣裙"}))
print(res.json())
