#
# Copyright 2023 DMetaSoul
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

PORT=8098
if [ -z "$PORT" ]; then
    echo "usage: sh start.sh <port>"
    exit
fi
if [ ! -f ".env" ]; then
    echo "Please create a .env file"
    exit
fi

export $(grep -v '^#' .env | xargs)
lsof -t -i:${PORT} | xargs kill -9 > /dev/null 2>&1
nohup uvicorn main:app --host=127.0.0.1 --port=${PORT} > start.log 2>&1 &
