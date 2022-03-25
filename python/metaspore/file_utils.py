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

def file_exists(url):
    import os
    from .s3_utils import s3_file_exists
    if url.startswith('s3://') or url.startswith('s3a://'):
        return s3_file_exists(url)
    else:
        return os.path.isfile(url)

def dir_exists(url):
    import os
    from .s3_utils import get_s3_dir_size
    if url.startswith('s3://') or url.startswith('s3a://'):
        return get_s3_dir_size(url) > 0
    else:
        return os.path.isdir(url)

def delete_dir(url):
    import os
    import shutil
    from .s3_utils import delete_s3_dir
    if url.startswith('s3://') or url.startswith('s3a://'):
        delete_s3_dir(url)
    else:
        if os.path.isdir(url):
            shutil.rmtree(url)

def delete_file(url):
    import os
    from .s3_utils import delete_s3_file
    if url.startswith('s3://') or url.startswith('s3a://'):
        delete_s3_file(url)
    else:
        if os.path.isfile(url):
            os.remove(url)

def copy_dir(src_url, dst_url):
    import shutil
    from .s3_utils import copy_s3_dir
    from .s3_utils import download_s3_dir
    from .s3_utils import upload_s3_dir
    if src_url.startswith('s3://') or src_url.startswith('s3a://'):
        if dst_url.startswith('s3://') or dst_url.startswith('s3a://'):
            copy_s3_dir(src_url, dst_url)
        else:
            download_s3_dir(src_url, dst_url)
    else:
        if dst_url.startswith('s3://') or dst_url.startswith('s3a://'):
            upload_s3_dir(src_url, dst_url)
        else:
            shutil.copytree(src_url, dst_url)
