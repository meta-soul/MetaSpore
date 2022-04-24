import os
import sys
import time
import argparse
import requests
from flask import Flask, jsonify, request, url_for
app = Flask(__name__)

from infer import TextEncoderInference

def http_get(url, path):
    """
    Downloads a URL to a given path on disc
    """
    if os.path.dirname(path) != '':
        os.makedirs(os.path.dirname(path), exist_ok=True)

    req = requests.get(url, stream=True)
    if req.status_code != 200:
        print("Exception when trying to download {}. Response {}".format(url, req.status_code), file=sys.stderr)
        req.raise_for_status()
        return

    download_filepath = path+"_part"
    with open(download_filepath, "wb") as file_binary:
        content_length = req.headers.get('Content-Length')
        total = int(content_length) if content_length is not None else None
        #progress = tqdm(unit="B", total=total, unit_scale=True)
        for chunk in req.iter_content(chunk_size=1024):
            if chunk: # filter out keep-alive new chunks
                #progress.update(len(chunk))
                file_binary.write(chunk)
    #progress.close()
    os.rename(download_filepath, path)

#assert os.path.isfile(os.environ.get('ONNX_CONF_PATH', '')), 'You must give the exported onnx json config file!'
#onnx_config = os.environ['ONNX_CONF_PATH']
#print(f"Loading model from {onnx_config}...")
#start_time = time.time()
#ort_encoder = TextEncoderInference.create_from_config(onnx_config, device='cpu')
#print("Loading model done using {}ms!".format((time.time()-start_time)*1000))

ort_encoder_map = {}

@app.route('/list', methods=['GET'])
def list_model():
    res = {'errno': 0, 'msg': 'ok', 'data': {}}
    for model in ort_encoder_map:
        res['data'][model] = [k for k in ort_encoder_map[model]]
    return jsonify(res)

@app.route('/push/<model>', methods=['GET'])
def push(model):
    res = {'errno': 0, 'msg': 'ok', 'data': {}}
    export_path = request.args.get('path', '')
    model_tag = request.args.get('tag', 'v1')
    model_tag = model_tag if model_tag else 'v1'
    device = request.args.get('device', 'cpu')
    config_path = os.path.join(export_path, 'onnx_config.json')
    if not model:
        res['errno'] = -1
        res['msg'] = 'model name is empty!'
        return jsonify(res)
    if not export_path or not os.path.isfile(config_path):
        res['errno'] = -1
        res['msg'] = 'invalid onnx export path!'
        return jsonify(res)
    if model not in ort_encoder_map:
        ort_encoder_map[model] = {}
    try:
        start_time = time.time()
        ort_encoder_map[model][model_tag] = TextEncoderInference.create_from_config(config_path, device=device)
    except Exception as e:
        res['errno'] = -1
        res['msg'] = 'load occur error: {}!'.format(e)
        return jsonify(res)
    res['msg'] = 'model push success!'
    base_url = os.path.dirname(os.path.dirname(request.base_url))
    res['data']['urls'] = {
        'embedding': base_url + url_for('embedding', model=model, tag=model_tag),
        'similarity': base_url + url_for('similarity', model=model, tag=model_tag)
    }
    return jsonify(res)


@app.route('/embedding/<model>/<tag>', methods=['GET'])
def embedding(model, tag):
    res = {'errno': 0, 'msg': 'ok', 'data': []}

    ort_encoder = ort_encoder_map.get(model, {}).get(tag, None)
    if ort_encoder is None:
        res['errno'] = -1
        res['msg'] = 'invalid model or tag'
        return jsonify(res)

    text = request.args.get('text', '')
    splitter = request.args.get('splitter', '')
    if not splitter:
        texts = [text]
    else:
        texts = text.split(splitter)
    res['data'] = ort_encoder(texts).tolist()
    return jsonify(res)

@app.route('/similarity/<model>/<tag>', methods=['GET'])
def similarity(model, tag):
    res = {'errno': 0, 'msg': 'ok', 'data': []}

    tag = request.args.get('tag', 'v1')
    ort_encoder = ort_encoder_map.get(model, {}).get(tag, None)
    if ort_encoder is None:
        res['errno'] = -1
        res['msg'] = 'invalid model or tag'
        return jsonify(res)

    text1 = request.args.get('text1', '')
    text2 = request.args.get('text2', '')
    splitter = request.args.get('splitter', '')
    if not splitter:
        texts1 = [text1]
        texts2 = [text2]
    else:
        texts1 = text1.split(splitter)
        texts2 = text2.split(splitter)
    size = min(len(texts1), len(texts2))
    texts1, texts2 = texts1[:size], texts2[:size]
    from sklearn.metrics.pairwise import cosine_similarity,cosine_distances
    embs1 = ort_encoder(texts1)
    embs2 = ort_encoder(texts2)
    mat = cosine_similarity(embs1, embs2).tolist()
    scores = []
    for i in range(len(mat)):
        scores.append(mat[i][i])
    res['data'] = scores
    return jsonify(res)

def shutdown_server():
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError('Not running with the Werkzeug Server')
    func()

@app.get('/shutdown')
def shutdown():
    shutdown_server()
    return 'Server shutting down...'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--action", choices=["startup", "shutdown", "push", "list"], required=True) 
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", default=9000, type=int)
    parser.add_argument("--model", default="")
    parser.add_argument("--tag", default="")
    parser.add_argument("--onnx-path", default="")
    args = parser.parse_args()

    base_url = 'http://{}:{}'.format(args.host, args.port)

    if args.action == "startup":
        app.run(host='0.0.0.0', port=9000)
    elif args.action == "shutdown":
        r = requests.get('{}/shutdown'.format(base_url))
        print(r.text)
    elif args.action == "push":
        url = '{}/push/{}'.format(base_url, args.model)
        r = requests.get(url, params={'path': args.onnx_path, 'tag': args.tag})
        print(r.json())
        #print('embedding url: {}'.format('{}/embedding/{}/{}'.format(base_url, args.model, args.tag)))
    elif args.action == "list":
        r = requests.get('{}/list'.format(base_url))
        print(r.json())
    else:
        print('invalid action!!!')
