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

def use_s3(url):
    return url.replace('s3a://', 's3://')

def use_s3a(url):
    return url.replace('s3://', 's3a://')

def is_url(string):
    if string.startswith('s3://') or string.startswith('s3a://'):
        return True
    if string.startswith('file://'):
        return True
    if string.startswith('./') or string.startswith('/'):
        return True
    return False
