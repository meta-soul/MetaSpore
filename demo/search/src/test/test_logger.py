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

import logging

log_file = 'log.txt'
log_level = logging.DEBUG

logger = logging.getLogger(__name__)
logger.setLevel(log_level)

file_handler = logging.FileHandler(filename=log_file)
stream_handler = logging.StreamHandler()
formatter = logging.Formatter(
    '%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d,%H:%M:%S')
file_handler.setFormatter(formatter)
stream_handler.setFormatter(formatter)
file_handler.setLevel(log_level)
stream_handler.setLevel(log_level)
logger.addHandler(file_handler)
logger.addHandler(stream_handler)

logger.info('abc')
logger.debug('debug')
logger.error('error')
