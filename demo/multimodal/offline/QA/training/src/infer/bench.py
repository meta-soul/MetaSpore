import os
import json
import time
import random
import argparse

# ignore warnings
import warnings
warnings.filterwarnings("ignore")
import logging
logging.getLogger().setLevel(logging.ERROR)

from modeling import TransformerEncoder
from infer import TextEncoderInference


parser = argparse.ArgumentParser()
parser.add_argument("--model-name", required=True, help="Torch model's name or path")
parser.add_argument("--onnx-path", required=True, help="ONNX model's export path")
parser.add_argument("--input-file", default="", help="The performance benchmark corpus file.")
parser.add_argument("--batch-size", type=int, default=1, help="The batch size.")
parser.add_argument("--n", type=int, default=1000, help="The number of model running.")
parser.add_argument("--device", default="cpu")
args = parser.parse_args()

texts = ['hello world!']
if args.input_file:
    texts = [line.strip() for line in open(args.input_file, 'r', encoding='utf8')]
if len(texts) < args.batch_size:
    texts = texts * args.batch_size
random.shuffle(texts)
texts = texts[:args.batch_size]

print('model', 'type', 'batch', 'device', 'latency', 'throughput', sep='\t')

base_model = args.model_name
torch_encoder = TransformerEncoder(base_model, device=args.device)
latency = []
for i in range(args.n):
    start_time = time.time()
    torch_encoder.encode(texts)
    latency.append((time.time() - start_time)*1000)
p1 = sum(latency)/len(latency)
p2 = (args.n*args.batch_size) / (sum(latency)/1000)
#print('\ttorch cost: {}s, process: {}samples'.format(sum(latency)/1000, args.n*args.batch_size))
#print('\ttorch latency: {}ms/batch, throughput: {}samples/s'.format(p1, p2))
print(base_model, 'torch', args.batch_size, args.device, p1, p2, sep='\t')

ort_encoder = TextEncoderInference.create_from_config(os.path.join(args.onnx_path, 'onnx_config.json'), device=args.device)
#print('model: {}'.format(base_model))
latency = []
for i in range(args.n):
    start_time = time.time()
    ort_encoder(texts)
    latency.append((time.time() - start_time)*1000)
p1 = sum(latency)/len(latency)
p2 = (args.n*args.batch_size) / (sum(latency)/1000)
#print('\tonnx cost: {}s, process: {}samples'.format(sum(latency)/1000, args.n*args.batch_size))
#print('\tonnx latency: {}ms/batch, throughput: {}samples/s'.format(p1, p2))
print(base_model, 'onnx', args.batch_size, args.device, p1, p2, sep='\t')

