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
import torch
import numpy as np
from tqdm import tqdm

from modeling import TransformerCrossEncoder

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model',
        required=True,
        type=str,
        help='The pretrained model name or path.'
    )
    parser.add_argument(
        '--input-file',
        required=True,
        type=str,
        help='The query and passage pair data.'
    )
    parser.add_argument(
        '--output-file',
        required=True,
        type=str,
        help='The score output file.'
    )
    parser.add_argument(
        '--input-q-i',
        type=int,
        default=2,
        help='The column index of query in the input file.'
    )
    parser.add_argument(
        '--input-p-i',
        type=int,
        default=3,
        help='The column index of passage in the input file.'
    )
    parser.add_argument(
        '--num-labels',
        type=int,
        default=2
    )
    parser.add_argument(
        '--task-type',
        type=str,
        default='multiclass'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=16
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu'
    )
    return parser.parse_args()

def batchify(it, batch_size=256):
    batch = []
    for x in it:
        batch.append(x)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch

def main():
    args = parse_args()
    model = TransformerCrossEncoder.load_pretrained(args.model,
        task_type=args.task_type, num_labels=args.num_labels)
    model.to(args.device)

    sents = []
    with open(args.input_file, 'r', encoding='utf8') as fin:
        for line in fin:
            line = line.strip('\r\n')
            if not line:
                continue
            fields = line.split('\t')
            query, passage = fields[args.input_q_i], fields[args.input_p_i]
            sents.append([query, passage])

    scores = []
    for batch in tqdm(list(batchify(sents, args.batch_size*10))):
        with torch.no_grad():
            s = model.predict(batch, batch_size=args.batch_size)
        scores.append(s)
    scores = np.concatenate(scores)

    with open(args.output_file, 'w', encoding='utf8') as fout:
        for score in scores:
            fout.write(f"{score}\n")

if __name__ == '__main__':
    main()
