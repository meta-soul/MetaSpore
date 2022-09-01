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

import collections
_cached_s3_config = None
_cached_s3fs_config = None
_cached_s3fs = None
_s3_config_fields = 'aws_region', 'aws_endpoint', 'aws_access_key_id', 'aws_secret_access_key'
S3Config = collections.namedtuple('S3Config', _s3_config_fields)

def get_s3_config():
    import os
    import configparser
    global _cached_s3_config
    if _cached_s3_config is not None:
        return _cached_s3_config
    aws_region = None
    aws_endpoint = None
    aws_access_key_id = None
    aws_secret_access_key = None
    config_file_path = os.path.expanduser('~/.aws/config')
    credentials_file_path = os.path.expanduser('~/.aws/credentials')
    parser = configparser.ConfigParser()
    parser.read(config_file_path)
    try:
        aws_region = parser['default']['region']
    except KeyError:
        pass
    try:
        s3_value = parser['default']['s3']
    except KeyError:
        s3_value = None
    if s3_value is not None:
        kv_config = dict()
        for line in s3_value.strip().splitlines():
            key, _, value = line.partition('=')
            kv_config[key.strip()] = value.strip()
        aws_endpoint = kv_config.get('endpoint_url')
    parser = configparser.ConfigParser()
    parser.read(credentials_file_path)
    try:
        aws_access_key_id = parser['default']['aws_access_key_id']
    except KeyError:
        pass
    try:
        aws_secret_access_key = parser['default']['aws_secret_access_key']
    except KeyError:
        pass
    region = os.environ.get('AWS_REGION')
    if region:
        aws_region = region
    endpoint = os.environ.get('AWS_ENDPOINT')
    if endpoint:
        aws_endpoint = endpoint
    access_key_id = os.environ.get('AWS_ACCESS_KEY_ID')
    if access_key_id:
        aws_access_key_id = access_key_id
    secret_access_key = os.environ.get('AWS_SECRET_ACCESS_KEY')
    if secret_access_key:
        aws_secret_access_key = secret_access_key
    if aws_endpoint:
        if not aws_endpoint.startswith('http://') and not aws_endpoint.startswith('https://'):
            aws_endpoint = 'http://' + aws_endpoint
    config = S3Config(aws_region, aws_endpoint, aws_access_key_id, aws_secret_access_key)
    if config.aws_region:
        os.environ['AWS_REGION'] = config.aws_region
    else:
        os.environ.unsetenv('AWS_REGION')
    if config.aws_endpoint:
        os.environ['AWS_ENDPOINT'] = config.aws_endpoint
    else:
        os.environ.unsetenv('AWS_ENDPOINT')
    if config.aws_access_key_id:
        os.environ['AWS_ACCESS_KEY_ID'] = config.aws_access_key_id
    else:
        os.environ.unsetenv('AWS_ACCESS_KEY_ID')
    if config.aws_secret_access_key:
        os.environ['AWS_SECRET_ACCESS_KEY'] = config.aws_secret_access_key
    else:
        os.environ.unsetenv('AWS_SECRET_ACCESS_KEY')
    _cached_s3_config = config
    return config

def get_s3fs_config():
    global _cached_s3fs_config
    if _cached_s3fs_config is not None:
        return _cached_s3fs_config
    config = get_s3_config()
    conf = dict(
        anon=False,
        key=config.aws_access_key_id,
        secret=config.aws_secret_access_key,
        config_kwargs={'proxies': {}, 'region_name': config.aws_region},
        client_kwargs={'endpoint_url': config.aws_endpoint}
    )
    _cached_s3fs_config = conf
    return conf

def get_s3fs():
    import fsspec
    global _cached_s3fs
    if _cached_s3fs is not None:
        return _cached_s3fs
    fs = fsspec.filesystem('s3', **get_s3fs_config())
    _cached_s3fs = fs
    return fs

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
    config = get_s3_config()
    return config.aws_endpoint

def get_aws_region():
    config = get_s3_config()
    return config.aws_region

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
