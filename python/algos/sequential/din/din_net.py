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
from ...layers import MLPLayer, LRLayer


class DIN(torch.nn.Module):
    def __init__(self,
                 use_wide=True,
                 use_deep=True,
                 din_embedding_dim=16,
                 wide_embedding_dim=16,
                 deep_embedding_dim=16,
                 din_column_name_path=None,
                 din_combine_schema_path=None,
                 wide_column_name_path=None,
                 wide_combine_schema_path=None,
                 deep_column_name_path=None,
                 deep_combine_schema_path=None,                 
                 din_attention_hidden_layers=[16, 8],
                 din_attention_hidden_activations='dice',
                 din_attention_batch_norm=True,
                 din_attention_dropout=0.1,
                 din_hidden_layers=[32, 16],
                 din_hidden_activations='dice',
                 din_hidden_batch_norm=True,
                 din_hidden_dropout=0.25,
                 seq_column_index_list=[1],
                 target_column_index_list = [2],
                 deep_hidden_units=[32, 16],
                 deep_hidden_activations='ReLU',
                 deep_hidden_dropout=0.2,
                 deep_hidden_batch_norm=True,
                 sparse_init_var=1e-2,
                 ftrl_l1=1.0,
                 ftrl_l2=120.0,
                 ftrl_alpha=0.5,
                 ftrl_beta=1.0
                ):
        super().__init__()
        
        self.use_wide = use_wide
        self.use_deep = use_deep
        # din_column_name_path = 's3://dmetasoul-bucket/xwb/demo/movielens/1m/rank/column_schema_DIN.txt'
        # din_combine_schema_path = 's3://dmetasoul-bucket/xwb/demo/movielens/1m/rank/combine_column_schema_DIN.txt'
        # wide_column_name_path = 's3://dmetasoul-bucket/xwb/demo/movielens/1m/rank/column_schema_DIN.txt'
        # wide_combine_schema_path = 's3://dmetasoul-bucket/xwb/demo/movielens/1m/rank/combine_column_schema_wide.txt'
        # deep_column_name_path = 's3://dmetasoul-bucket/xwb/demo/movielens/1m/rank/column_schema_DIN.txt'
        # deep_combine_schema_path = 's3://dmetasoul-bucket/xwb/demo/movielens/1m/rank/combine_column_schema_deep.txt'

        self.embedding_table = ms.EmbeddingLookup(din_embedding_dim, din_column_name_path, din_combine_schema_path)
        self.embedding_table.updater = ms.FTRLTensorUpdater(l1=ftrl_l1, l2=ftrl_l2, alpha = ftrl_alpha, beta=ftrl_beta)
        self.embedding_table.initializer = ms.NormalTensorInitializer(var=sparse_init_var)
        
        self.feature_nums = self.embedding_table.feature_count
        self.seq_column_index_list = seq_column_index_list
        self.target_column_index_list = target_column_index_list
        self.other_column_index_list = []
        for i in range(self.feature_nums):
            if i not in self.seq_column_index_list and i not in self.target_column_index_list:
                self.other_column_index_list.append(i)
        concat_feature_num = len(seq_column_index_list)
        self._attention = Attention(input_dim=din_embedding_dim * concat_feature_num,
                                    hidden_units=din_attention_hidden_layers,
                                    hidden_activations=din_attention_hidden_activations,
                                    dropout_rates=din_attention_dropout,
                                    batch_norm=din_attention_batch_norm)

        total_input_size = din_embedding_dim * self.feature_nums
        self.mlp = MLPLayer(input_dim=total_input_size,
                            output_dim=1,
                            hidden_units=din_hidden_layers,
                            hidden_activations=din_hidden_activations,
                            final_activation=None,
                            dropout_rates=din_hidden_dropout, 
                            batch_norm=din_hidden_batch_norm
                            )
    
        if self.use_wide:
            self.lr_sparse = ms.EmbeddingSumConcat(wide_embedding_dim, wide_column_name_path, wide_combine_schema_path)
            self.lr_sparse.updater = ms.FTRLTensorUpdater(l1=ftrl_l1, l2=ftrl_l2, alpha=ftrl_alpha, beta=ftrl_beta)
            self.lr_sparse.initializer = ms.NormalTensorInitializer(var=sparse_init_var)
            self.lr = LRLayer(wide_embedding_dim, self.lr_sparse.feature_count)
        ## nn layers
        if self.use_deep:
            self.deep_sparse = ms.EmbeddingSumConcat(deep_embedding_dim, deep_column_name_path, deep_combine_schema_path)
            self.deep_sparse.updater = ms.FTRLTensorUpdater(l1=ftrl_l1, l2=ftrl_l2, alpha=ftrl_alpha, beta=ftrl_beta)
            self.deep_sparse.initializer = ms.NormalTensorInitializer(var=sparse_init_var)        
            self.deep = MLPLayer(input_dim=self.deep_sparse.feature_count * deep_embedding_dim,
                                output_dim=1,
                                hidden_units=deep_hidden_units,
                                hidden_activations=deep_hidden_activations,
                                final_activation=None, 
                                dropout_rates=deep_hidden_dropout, 
                                batch_norm=deep_hidden_batch_norm
                                )

        self.final_activation = torch.nn.Sigmoid()
        
    def forward(self, x):
        x, offset = self.embedding_table(x)     
        column_nums = self.feature_nums
        # 利用offset计算取出每个feature的embedding
        x_reshape = [x[offset[i]:offset[i+1],:] for i in range(offset.shape[0]-1)]
        x_reshape.append(x[offset[offset.shape[0]-1]:x.shape[0],:])
        
        other_embedding = None
        for other_column_idx in self.other_column_index_list:
            other_column_embedding = x_reshape[other_column_idx::column_nums]
            other_column_embedding = torch.stack(other_column_embedding)
            other_column_embedding = other_column_embedding.squeeze()
            if other_embedding is None:
                other_embedding = other_column_embedding
            else:
                other_embedding = torch.cat((other_embedding, other_column_embedding), dim=1) 
            
        target_embedding = None
        for target_column_index in self.target_column_index_list:
            target_column_embedding = x_reshape[target_column_index::column_nums]
            target_column_embedding = torch.stack(target_column_embedding).squeeze()
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
        emb_concat = torch.cat((other_embedding, all_sum_pooling, target_embedding), dim=1)     
        din_out = self.mlp(emb_concat) 
        if self.use_wide:
            lr_feature_map = self.lr_sparse(x)
            lr_out = self.lr(lr_feature_map)
            din_out += lr_out
        if self.use_deep:
            nn_feature_map = self.deep_sparse(x)
            deep_out = self.deep(nn_feature_map)
            din_out += deep_out
        
        return self.final_activation(din_out)