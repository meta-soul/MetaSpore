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
import shutil
import logging
import tarfile
from concurrent import futures

import grpc
from grpc_reflection.v1alpha import reflection
import requests

from hf_preprocessor import hf_preprocessor_pb2
from hf_preprocessor import hf_preprocessor_pb2_grpc
from hf_preprocessor.hf_tokenizer import HfTokenizer

tokenizer_map = {}

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


class HfPreprocessor(hf_preprocessor_pb2_grpc.HfPreprocessorServicer):

    def __init__(self, tmp_dir, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tmp_dir = tmp_dir

    def HfTokenizer(self, request, context):
        model_key = request.model_name
        extras = {'status': '0', 'msg': 'ok'}
        payload = {}

        if model_key not in tokenizer_map:
            extras['status'] = '1'
            extras['msg'] = 'Invalid model key!'
            return hf_preprocessor_pb2.HfTokenizerResponse(payload=payload, extras=extras)

        try:
            outputs = tokenizer_map[model_key].predict(request.payload)
        except Exception as e:
            raise e
            extras['status'] = '1'
            extras['msg'] = 'Failed: {}'.format(e)
            return hf_preprocessor_pb2.HfTokenizerResponse(payload=payload, extras=extras)

        payload.update(outputs)
        return hf_preprocessor_pb2.HfTokenizerResponse(payload=payload, extras=extras)

    def HfTokenizerPush(self, request, context):
        model_key, model_url = request.model_name, request.model_url
        model_name = os.path.basename(model_url)
        
        try:
            if os.path.isfile(model_url):
                tmp_model_path = model_url
            else:
                tmp_model_path = os.path.join(self.tmp_dir, model_name)
                http_get(model_url, tmp_model_path)

            with tarfile.open(tmp_model_path, "r:gz") as tar:
                members = tar.getmembers()
                tarname = members[0].name

                extract_dir = os.path.join(self.tmp_dir, tarname)
                shutil.rmtree(extract_dir, ignore_errors=True)
                def is_within_directory(directory, target):
                    
                    abs_directory = os.path.abspath(directory)
                    abs_target = os.path.abspath(target)
                
                    prefix = os.path.commonprefix([abs_directory, abs_target])
                    
                    return prefix == abs_directory
                
                def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
                
                    for member in tar.getmembers():
                        member_path = os.path.join(path, member.name)
                        if not is_within_directory(path, member_path):
                            raise Exception("Attempted Path Traversal in Tar File")
                
                    tar.extractall(path, members, numeric_owner=numeric_owner) 
                    
                
                safe_extract(tar, path=self.tmp_dir, members=members)

                tokenizer_map[model_key] = HfTokenizer.load(extract_dir)
                shutil.rmtree(extract_dir, ignore_errors=True)

            os.remove(tmp_model_path)
        except Exception as e:
            return hf_preprocessor_pb2.HfTokenizerPushResponse(status=1, msg='Failed: {}'.format(e))

        return hf_preprocessor_pb2.HfTokenizerPushResponse(status=0, msg='ok')

def serve(port, tmp_dir, max_workers):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    service = HfPreprocessor(tmp_dir)
    hf_preprocessor_pb2_grpc.add_HfPreprocessorServicer_to_server(service, server)
    # add service reflection
    SERVICE_NAMES = (
        hf_preprocessor_pb2.DESCRIPTOR.services_by_name['HfPreprocessor'].full_name,
        reflection.SERVICE_NAME,
    )
    reflection.enable_server_reflection(SERVICE_NAMES, server)
    server.add_insecure_port(f'[::]:{port}')
    server.start()
    server.wait_for_termination()


if __name__ == '__main__':
    port, tmp_dir, num_workers = sys.argv[1:4]

    logging.basicConfig()
    os.makedirs(tmp_dir, exist_ok=True)
    serve(port, tmp_dir, max_workers=int(num_workers))
