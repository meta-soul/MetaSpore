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

        self._attention = Attention(input_dim=embedding_dim,
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
        # print("offset: ", offset)
        
        column_nums = self.feature_nums
        # 利用offset计算取出每个feature的embedding
        x_reshape = [x[offset[i]:offset[i+1],:] for i in range(offset.shape[0]-1)]
        x_reshape.append(x[offset[offset.shape[0]-1]:x.shape[0],:])
        # print("x_reshape: ", len(x_reshape)) # batch_size * feature数量
        user_column_index = self.user_column_index
        user_embedding = x_reshape[user_column_index::column_nums]
        # print("user_embdding:", len(user_embedding)) # 100
        user_embedding = torch.nn.utils.rnn.pad_sequence(user_embedding, batch_first = True) # [B 1 H]
        user_embedding = user_embedding.squeeze() 
        # print("user_embedding.shape: ", user_embedding.shape) # [B H]
            
        # attention group
        # seq_column_index_list = [1, 3]
        # target_column_index_list = [2, 4]
        all_sum_pooling=None
        for seq_column_index, target_column_index in zip(self.seq_column_index_list, self.target_column_index_list):
            # 计算item_seq_length
            item_seq_length = [offset[i] - offset[i-1] for i in range(seq_column_index+1, offset.shape[0], column_nums)]
            item_seq_length = torch.tensor(item_seq_length) # 维度为[B]
            # print("item_seq_length: ", item_seq_length)
        
            # 长度为100的列表，列表中每个元素就是每个样本的item_seq对应的tensor 
            item_seq_embedding = x_reshape[seq_column_index::column_nums]
            # print(item_seq_embedding)
            item_seq_embedding = torch.nn.utils.rnn.pad_sequence(item_seq_embedding, batch_first=True)
            # print("item_seq_embdding.shape: ", item_seq_embedding.shape) # [B T H]

            target_item_embedding = x_reshape[target_column_index::column_nums]
            # print("target_item_embedding:", len(target_item_embedding)) # B
            target_item_embedding = torch.nn.utils.rnn.pad_sequence(target_item_embedding, batch_first = True)
            target_item_embedding = target_item_embedding.squeeze() 
            # print("target_item_embedding.shape: ", target_item_embedding.shape) # [B H]
            sum_pooling = self._attention(target_item_embedding, item_seq_embedding, item_seq_length) 
            # print(sum_pooling.shape)
            if all_sum_pooling == None:
                all_sum_pooling = torch.cat((sum_pooling, target_item_embedding), dim=1)
            else:
                all_sum_pooling = torch.cat((all_sum_pooling, sum_pooling, target_item_embedding), dim=1)
            
        # print("all_sum_pooling.shape: ", all_sum_pooling.shape)
        emb_concat = torch.cat((user_embedding, all_sum_pooling), dim=1) 
        # print("emb_concat.shape: ", emb_concat.shape) 
        output = self.mlp(emb_concat) # [B  1]
        # print("output.shape: ", output.shape)
        return output
    