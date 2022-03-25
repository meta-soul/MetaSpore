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

def parse_s3_url(s3_url):
    from urllib.parse import urlparse
    r = urlparse(s3_url, allow_fragments=False)
    if r.scheme not in ('s3', 's3a'):
        message = "invalid s3 url: %r" % (s3_url,)
        raise ValueError(message)
    path = r.path.lstrip('/')
    return r.netloc, path

def parse_s3_dir_url(s3_url):
    bucket, path = parse_s3_url(s3_url)
    if not path.endswith('/'):
        path += '/'
    return bucket, path

def get_aws_endpoint():
    import os
    endpoint = os.environ.get('AWS_ENDPOINT')
    if endpoint is not None:
        if not endpoint.startswith('http://') and not endpoint.startswith('https://'):
            endpoint = 'http://' + endpoint
    return endpoint

def get_aws_region():
    import os
    region = os.environ.get('AWS_REGION')
    return region

def get_s3_client():
    import boto3
    endpoint = get_aws_endpoint()
    region = get_aws_region()
    s3 = boto3.client('s3', endpoint_url=endpoint, region_name = region)
    return s3

def get_s3_resource():
    import boto3
    endpoint = get_aws_endpoint()
    region = get_aws_region()
    s3 = boto3.resource('s3', endpoint_url=endpoint, region_name = region)
    return s3

def get_s3_dir_size(dir_path):
    bucket, path = parse_s3_dir_url(dir_path)
    s3 = get_s3_client()
    objs = s3.list_objects(Bucket=bucket, Prefix=path)
    size = 0
    if 'Contents' in objs:
        for obj in objs['Contents']:
            size += obj['Size']
    return size

def s3_file_exists(file_path):
    bucket, path = parse_s3_url(file_path)
    s3 = get_s3_client()
    try:
        s3.head_object(Bucket=bucket, Key=path)
    except:
        return False
    else:
        return True

def delete_s3_dir(dir_path):
    bucket, path = parse_s3_dir_url(dir_path)
    s3 = get_s3_resource()
    for object in s3.Bucket(bucket).objects.filter(Prefix=path).all():
        object.delete()

def delete_s3_file(file_path):
    bucket, path = parse_s3_url(file_path)
    s3 = get_s3_resource()
    s3.Object(bucket, path).delete()

def copy_s3_dir(src_dir_path, dst_dir_path):
    src_bucket, src_dir = parse_s3_dir_url(src_dir_path)
    dst_bucket, dst_dir = parse_s3_dir_url(dst_dir_path)
    s3 = get_s3_resource()
    bucket = s3.Bucket(dst_bucket)
    for item in s3.Bucket(src_bucket).objects.filter(Prefix=src_dir):
        src = { 'Bucket' : item.bucket_name, 'Key' : item.key }
        dst = dst_dir + item.key[len(src_dir):]
        bucket.copy(src, dst)

def download_s3_dir(src_dir_path, dst_dir_path):
    import os
    from . import _metaspore
    src_bucket, src_dir = parse_s3_dir_url(src_dir_path)
    s3 = get_s3_resource()
    bucket = s3.Bucket(src_bucket)
    for item in bucket.objects.filter(Prefix=src_dir):
        src = item.key
        dst = os.path.join(dst_dir_path, item.key[len(src_dir):])
        _metaspore.ensure_local_directory(os.path.dirname(dst))
        bucket.download_file(src, dst)

def upload_s3_dir(src_dir_path, dst_dir_path):
    import os
    if not src_dir_path.endswith('/'):
        src_dir_path += '/'
    dst_bucket, dst_dir = parse_s3_dir_url(dst_dir_path)
    s3 = get_s3_resource()
    bucket = s3.Bucket(dst_bucket)
    for dirpath, dirnames, filenames in os.walk(src_dir_path):
        for filename in filenames:
            src = os.path.join(dirpath, filename)
            dst = dst_dir + src[len(src_dir_path):]
            bucket.upload_file(src, dst)
