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
import json
import argparse

from tqdm import tqdm


def load_baike_jsonline(file, start_id=0):
    with open(file, 'r', encoding='utf8') as fin:
        for line in tqdm(fin):
            line = line.strip('\r\n')
            if not line:
                continue
            item = json.loads(line)
            question = ''
            answer = ''
            q_keys = ['question', 'title']
            for k in q_keys:
                question = item.get(k, '')
                if question:
                    break
            a_keys = ['answer']
            for k in a_keys:
                answer = item.get(k, '')
                if answer:
                    break
            question = question.replace('\r\n', ' ').replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
            answer = answer.replace('\r\n', '<br/>').replace('\n', '<br/>').replace('\r', '<br/>').replace('\t', ' ')
            if not question or not answer:
                continue
            category = [cate.strip() for cate in item.get('category', '').split('-')]
            doc = {
                'id': str(start_id),
                'question': question,
                'answer': answer,
                'category': category
            }
            start_id += 1
            yield doc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-jsonline", required=True
    )
    parser.add_argument(
        "--output-jsonline", required=True
    )
    parser.add_argument(
        "--start-id", default=0, type=int
    )
    args = parser.parse_args()

    with open(args.output_jsonline, 'w', encoding='utf8') as fout:
        for doc in load_baike_jsonline(args.input_jsonline, args.start_id):
            fout.write('{}\n'.format(json.dumps(doc, ensure_ascii=False)))

    print("Preprocess done!")
