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

# Text completion
#curl -H "Content-Type: application/json" -X POST -d '{"inputs":"连衣裙","args":"{\"num_sentences\":3}"}' http://127.0.0.1:8098/api/infer/text-completion -o out.completion.json

# Text to Image
#curl -H "Content-Type: application/json" -X POST -d '{"inputs":"连衣裙"}' http://127.0.0.1:8098/api/infer/text-to-image -o out.t2i.json

# Text to Text
#curl -H "Content-Type: application/json" -X POST -d '{"inputs":"连衣裙","args":"{\"num_sentences\":3}"}' http://127.0.0.1:8098/api/infer/text-to-text -o out.t2t.json

# Text translation
curl -H "Content-Type: application/json" -X POST -d '{"inputs":"连衣裙","args":"{\"num_sentences\":3}"}' http://127.0.0.1:8098/api/infer/text-translation?model_type=zh2en -o out.translation.json
