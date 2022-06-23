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

LOG_FORMAT = '%(asctime)s - %(message)s'
DATE_FORMAT='%Y-%m-%d %H:%M:%S'
DEFAULT_LEVEL = 'info'
LEVELS = {
    'debug':logging.DEBUG,
    'info':logging.INFO,
    'warning':logging.WARNING,
    'error':logging.ERROR,
    'critical':logging.CRITICAL}

def start_logging(level=DEFAULT_LEVEL, **params):
    logging.basicConfig(level=LEVELS[level], datefmt=DATE_FORMAT, format=LOG_FORMAT)
    return logging.getLogger(__name__)