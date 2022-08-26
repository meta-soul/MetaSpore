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

import torch 
import metaspore as ms
from ...layers import MLPLayer, Attention


class DIN(torch.nn.Module):
    def __init__(self,
                 feature_nums=5,
                 embedding_dim=16,
                 column_name_path=None,
                 combine_schema_path=None,
                 sparse_init_var=1e-2,
                 attention_hidden_layers=[16, 8],
                 attention_hidden_activations='dice',
                 attention_batch_norm=True,
                 attention_dropout=0.0,
                 mlp_hidden_layers=[32,16],
                 mlp_hidden_activations='dice',
                 mlp_batch_norm=True,
                 mlp_dropout=0.25,
                 user_column_index=0,
                 seq_column_index_list=[1, 3],
                 target_column_index_list = [2, 4],
                 ftrl_l1=1.0,
                 ftrl_l2=120.0,
                 ftrl_alpha=0.5,
                 ftrl_beta=1.0
                ):
        super().__init__()

        self._embedding_table = ms.EmbeddingLookup(embedding_dim, column_name_path, combine_schema_path)
        self._embedding_table.updater = ms.FTRLTensorUpdater(l1=ftrl_l1, l2=ftrl_l2, alpha = ftrl_alpha, beta=ftrl_beta)
        self._embedding_table.initializer = ms.NormalTensorInitializer(var=sparse_init_var)
        
        self.feature_nums = feature_nums
        self.user_column_index = user_column_index
        self.seq_column_index_list = seq_column_index_list
        self.target_column_index_list = target_column_index_list
        
        concat_feature_num = len(seq_column_index_list)
        self._attention = Attention(input_dim=embedding_dim * concat_feature_num,
                                    hidden_units=attention_hidden_layers,
                                    hidden_activations=attention_hidden_activations,
                                    dropout_rates=attention_dropout,
                                    batch_norm=attention_batch_norm)

        total_input_size = embedding_dim * feature_nums
        self.mlp = MLPLayer(input_dim=total_input_size,
                            output_dim=1,
                            final_activation="sigmoid",
                            hidden_units=mlp_hidden_layers,
                            dropout_rates=mlp_dropout, 
                            batch_norm=mlp_batch_norm, 
                            hidden_activations=mlp_hidden_activations)
        
    def forward(self, x):
        x, offset = self._embedding_table(x)     
        column_nums = self.feature_nums
        # 利用offset计算取出每个feature的embedding
        x_reshape = [x[offset[i]:offset[i+1],:] for i in range(offset.shape[0]-1)]
        x_reshape.append(x[offset[offset.shape[0]-1]:x.shape[0],:])
        user_column_index = self.user_column_index
        user_embedding = x_reshape[user_column_index::column_nums]
        user_embedding = torch.stack(user_embedding).squeeze()
        user_embedding = user_embedding.squeeze() 
            
        target_embedding = None
        for target_column_index in self.target_column_index_list:
            target_column_embedding = x_reshape[target_column_index::column_nums]
            target_column_embedding = torch.stack(target_column_embedding) 
            target_column_embedding = target_column_embedding.squeeze()
            if target_embedding is None:
                target_embedding = target_column_embedding
            else:
                target_embedding = torch.cat((target_embedding, target_column_embedding), dim=1) 
        seq_embedding = None
        for seq_column_index in self.seq_column_index_list:
            if seq_embedding is None:
                item_seq_length = [offset[i] - offset[i-1] for i in range(seq_column_index+1, offset.shape[0], column_nums)]
                item_seq_length = torch.tensor(item_seq_length)
            item_seq_embedding = x_reshape[seq_column_index::column_nums]         
            item_seq_embedding = torch.nn.utils.rnn.pad_sequence(item_seq_embedding, batch_first=True)
            if seq_embedding is None:
                seq_embedding = item_seq_embedding
            else:
                seq_embedding = torch.cat((seq_embedding, item_seq_embedding), dim=2) 
        all_sum_pooling = self._attention(target_embedding, seq_embedding, item_seq_length)         
        emb_concat = torch.cat((user_embedding, all_sum_pooling, target_embedding), dim=1)     
        output = self.mlp(emb_concat) 
        
        return output
    