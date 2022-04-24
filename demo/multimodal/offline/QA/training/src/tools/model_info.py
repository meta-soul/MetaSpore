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

import sys
from transformers import AutoTokenizer, AutoModel, AutoConfig

model_dir = sys.argv[1]
config = AutoConfig.from_pretrained(model_dir)
model = AutoModel.from_pretrained(model_dir)

num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#num_params = sum(p.numel() for p in model.parameters())
info = {
    'backbone': config.model_type,
    'num_layers': config.num_hidden_layers, 
    'num_heads': config.num_attention_heads,
    'hidden_dim': config.hidden_size,
    'ffnn_dim': config.intermediate_size,
    'max_seq_len': config.max_position_embeddings,
    'vocab_size': config.vocab_size,
    'num_params': '{}M'.format(int(num_params/1000000))
}
for k, v in info.items():
    print(k, v, sep='\t')

