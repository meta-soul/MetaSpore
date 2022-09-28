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
from ...layers import MLPLayer, DIEN_DIN_AttLayer, LRLayer



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
                 # sequence column din_seq_column_index_list[i] will do attention with the target column din_target_column_index_list[i]
                 din_seq_column_index_list=[1, 3],
                 din_target_column_index_list = [2, 4],
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

        try:
            if len(din_seq_column_index_list) != len(din_target_column_index_list):
                raise Exception("din_seq_column_index_list and din_target_column_index_list must have same length")
        except Exception as e:
            print("error：",repr(e))

        self.use_wide = use_wide
        self.use_deep = use_deep

        self.embedding_table = ms.EmbeddingLookup(din_embedding_dim, din_column_name_path, din_combine_schema_path)
        self.embedding_table.updater = ms.FTRLTensorUpdater(l1=ftrl_l1, l2=ftrl_l2, alpha = ftrl_alpha, beta=ftrl_beta)
        self.embedding_table.initializer = ms.NormalTensorInitializer(var=sparse_init_var)

        self.feature_nums = self.embedding_table.feature_count
        self.seq_column_index_list = din_seq_column_index_list
        self.target_column_index_list = din_target_column_index_list
        self.other_column_index_list = []
        for i in range(self.feature_nums):
            if i not in self.seq_column_index_list and i not in self.target_column_index_list:
                self.other_column_index_list.append(i)
        concat_feature_num = len(din_seq_column_index_list)
        self.DIN_attention = DIEN_DIN_AttLayer(input_dim=din_embedding_dim * concat_feature_num,
                                    att_hidden_size=din_attention_hidden_layers,
                                    att_activation=din_attention_hidden_activations,
                                    att_dropout=din_attention_dropout,
                                    use_att_bn=din_attention_batch_norm)

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

    def get_seq_column_embedding(self, seq_column_index_list, x_reshape, column_nums):
        all_column_embedding = []
        for column_index in seq_column_index_list:
            column_embedding = x_reshape[column_index::column_nums]
            column_embedding = torch.nn.utils.rnn.pad_sequence(column_embedding, batch_first=True)
            all_column_embedding.append(column_embedding)
        all_column_embedding = torch.cat(all_column_embedding, dim=2)
        return all_column_embedding

    def get_non_seq_column_embedding(self, non_seq_column_index_list, x_reshape, column_nums):
        all_column_embedding = []
        for column_index in non_seq_column_index_list:
            column_embedding = x_reshape[column_index::column_nums]
            column_embedding = torch.stack(column_embedding).squeeze(1)
            all_column_embedding.append(column_embedding)
        all_column_embedding = torch.cat(all_column_embedding, dim=1)
        return all_column_embedding

    def get_field_embedding_list(self, x, offset):
        x_reshape = [x[offset[i]:offset[i+1],:] for i in range(offset.shape[0]-1)]
        x_reshape.append(x[offset[offset.shape[0]-1]:x.shape[0],:])
        return x_reshape

    def get_seq_length(self, seq_column_index_list, x, offset, column_nums):
        start_idx = self.seq_column_index_list[0]+1
        item_seq_length = [offset[i] - offset[i-1] for i in range(start_idx, offset.shape[0], column_nums)]
        if start_idx == column_nums:
            item_seq_length.append(x.shape[0]-offset[-1])
        item_seq_length = torch.tensor(item_seq_length)
        return item_seq_length

    def forward(self, x):
        x, offset = self.embedding_table(x)
        x_reshape = self.get_field_embedding_list(x, offset)
        column_nums = self.feature_nums
        other_embedding = self.get_non_seq_column_embedding(self.other_column_index_list, x_reshape, column_nums)
        target_embedding = self.get_non_seq_column_embedding(self.target_column_index_list, x_reshape, column_nums)
        seq_embedding = self.get_seq_column_embedding(self.seq_column_index_list, x_reshape, column_nums)
        item_seq_length = self.get_seq_length(self.seq_column_index_list, x, offset, column_nums)
        all_sum_pooling = self.DIN_attention(target_embedding, seq_embedding, item_seq_length).squeeze(1)
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
