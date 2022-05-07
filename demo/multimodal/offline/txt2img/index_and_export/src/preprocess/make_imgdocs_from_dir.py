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
import glob
import argparse


def read_img_from_dir(img_dir):
    for currdir, dirs, files in os.walk(img_dir):
        for file in files:
            file_path = os.path.join(currdir, file)
            yield file, file_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--img-dir", required=True)
    parser.add_argument("--output-jsonline", required=True)
    parser.add_argument("--base-url", default="")
    parser.add_argument("--img-exts", default="jpg,jpeg,png,gif")
    args = parser.parse_args()

    ext_set = set(args.img_exts.split(','))
    base_url = args.base_url.strip('/') if args.base_url else ""

    i = 0
    doc_list = []
    for img_name, img_path in read_img_from_dir(args.img_dir):
        if '.' not in img_name or os.path.splitext(img_name)[1].lstrip('.').lower() not in ext_set:
            continue
        doc = {
            'id': i,
            'name': img_name,
            'image': img_path,
            'url': os.path.join(base_url, img_name) if base_url else '/{}'.format(img_name)
        }
        i += 1
        doc_list.append(doc)

    with open(args.output_jsonline, 'w', encoding='utf8') as fout:
        for doc in doc_list:
            print(json.dumps(doc, ensure_ascii=False), file=fout)
