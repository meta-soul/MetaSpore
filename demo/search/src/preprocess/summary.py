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

max_len = -1
if len(sys.argv) > 1:
    max_len = int(sys.argv[1])

def process(text, max_len=-1):
    text = text.strip()
    text = text.replace('\t', ' ').replace('\r', ' ').replace('\n', ' ')
    if max_len > 0:
        text = text[:max_len]
    return text

summary = {
    'num_queries': 0,
    'num_passages': 0,
    'avg_query_passages': 0,
    'avg_query_len': 0,
    'avg_passage_len': 0
}

for line in sys.stdin:
    item = json.loads(line.strip())
    question = item['question']
    paragraphs = [p['paragraph_text'] for p in item['answer_paragraphs']]

    if not question or not paragraphs:
        continue

    summary['num_queries'] += 1
    summary['num_passages'] += len(paragraphs)
    n = summary['num_queries']
    summary['avg_query_passages'] = (summary['avg_query_passages'] * (n-1) + len(paragraphs)) / n
    summary['avg_query_len'] = (summary['avg_query_len'] * (n-1) + len(question)) / n
    summary['avg_passage_len'] = (summary['avg_passage_len'] * (n-1) + sum([len(p) for p in paragraphs])/len(paragraphs)) / n

for name, n in summary.items():
    print(name, n, sep='\t')
