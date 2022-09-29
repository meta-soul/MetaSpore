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
from ...layers import MLPLayer, TransformerEncoder, LRLayer



class BST(torch.nn.Module):
    def __init__(self,
                 max_seq_length=10,
                 use_wide=False,
                 use_deep=False,
                 bst_embedding_dim=16,
                 wide_embedding_dim=16,
                 deep_embedding_dim=16,
                 bst_column_name_path=None,
                 bst_combine_schema_path=None,
                 wide_column_name_path=None,
                 wide_combine_schema_path=None,
                 deep_column_name_path=None,
                 deep_combine_schema_path=None,  
                 bst_trm_n_layers=2,
                 bst_trm_n_heads=2,
                 bst_trm_inner_size=256,
                 bst_trm_hidden_dropout=0.5,
                 bst_trm_attn_dropout=0.5,
                 bst_trm_hidden_act='gelu',
                 bst_hidden_layers=[32,16],
                 bst_hidden_activations='LeakyReLU',
                 bst_hidden_batch_norm=True,
                 bst_hidden_dropout=0.15,
                 bst_seq_column_index_list=[1,3],
                 bst_target_column_index_list = [2,4],
                 deep_hidden_units=[32,16],
                 deep_hidden_activations="ReLU",
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
        self.embedding_table = ms.EmbeddingLookup(bst_embedding_dim, bst_column_name_path, bst_combine_schema_path)
        self.embedding_table.updater = ms.FTRLTensorUpdater(l1=ftrl_l1, l2=ftrl_l2, alpha = ftrl_alpha, beta=ftrl_beta)
        self.embedding_table.initializer = ms.NormalTensorInitializer(var=sparse_init_var)
        
        self.feature_nums = self.embedding_table.feature_count
        self.seq_column_index_list = bst_seq_column_index_list
        self.target_column_index_list = bst_target_column_index_list
        self.other_column_index_list = []
        for i in range(self.feature_nums):
            if i not in self.seq_column_index_list and i not in self.target_column_index_list:
                self.other_column_index_list.append(i)
        
        self.max_seq_length = max_seq_length
        self.total_seq_length = max_seq_length + 1
        self.positional_embedding = torch.nn.Embedding(self.total_seq_length, bst_embedding_dim)   
            
        self.trm_layer = TransformerEncoder(n_layers=bst_trm_n_layers,
                                            n_heads=bst_trm_n_heads,
                                            hidden_size=(len(self.seq_column_index_list) + 1) * bst_embedding_dim,
                                            inner_size=bst_trm_inner_size,
                                            hidden_dropout_prob=bst_trm_hidden_dropout,
                                            attn_dropout_prob=bst_trm_attn_dropout,
                                            hidden_act=bst_trm_hidden_act)
        # other_embedding + seq_embedding + target_embedding
        self.trm_out_size = bst_embedding_dim * max_seq_length * (len(self.seq_column_index_list) + 1)
        mlp_input_size = bst_embedding_dim * len(self.other_column_index_list) + self.trm_out_size + bst_embedding_dim * (len(self.target_column_index_list) + 1)
        self.mlp = MLPLayer(input_dim=mlp_input_size,
                            output_dim=1,
                            final_activation=None,
                            hidden_units=bst_hidden_layers,
                            dropout_rates=bst_hidden_dropout, 
                            batch_norm=bst_hidden_batch_norm, 
                            hidden_activations=bst_hidden_activations)
        
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
        
    @staticmethod
    def get_seq_column_embedding(seq_column_index_list, x_reshape, column_nums):
        all_column_embedding = []
        for column_index in seq_column_index_list:
            column_embedding = x_reshape[column_index::column_nums]         
            column_embedding = torch.nn.utils.rnn.pad_sequence(column_embedding, batch_first=True)
            all_column_embedding.append(column_embedding)
        all_column_embedding = torch.cat(all_column_embedding, dim=2)
        return all_column_embedding
    
    @staticmethod
    def get_non_seq_column_embedding(non_seq_column_index_list, x_reshape, column_nums):
        all_column_embedding = []
        for column_index in non_seq_column_index_list:
            column_embedding = x_reshape[column_index::column_nums]
            column_embedding = torch.stack(column_embedding).squeeze(1)
            all_column_embedding.append(column_embedding)
        all_column_embedding = torch.cat(all_column_embedding, dim=1) 
        return all_column_embedding
    
    @staticmethod
    def get_field_embedding_list(x, offset):
        x_reshape = [x[offset[i]:offset[i+1],:] for i in range(offset.shape[0]-1)]
        x_reshape.append(x[offset[offset.shape[0]-1]:x.shape[0],:])
        return x_reshape
    
    def forward(self, x):
        x, offset = self.embedding_table(x)     
        x_reshape = BST.get_field_embedding_list(x, offset)
        column_nums = self.feature_nums
        other_embedding = BST.get_non_seq_column_embedding(self.other_column_index_list, x_reshape, column_nums)
        target_embedding = BST.get_non_seq_column_embedding(self.target_column_index_list, x_reshape, column_nums).unsqueeze(1)  
        seq_embedding = BST.get_seq_column_embedding(self.seq_column_index_list, x_reshape, column_nums) # [B T 2*H]
        pad_tensor = torch.zeros(seq_embedding.shape[0], self.max_seq_length-seq_embedding.shape[1], seq_embedding.shape[2])
        seq_embedding = torch.cat((seq_embedding, pad_tensor), dim=1)
        trm_seq_feature = torch.cat((seq_embedding, target_embedding), dim=1) 
        positional_embedding = self.positional_embedding.weight.unsqueeze(0).repeat(trm_seq_feature.shape[0], 1, 1)
        trm_input_feature = torch.cat((trm_seq_feature, positional_embedding), dim=2)
        transformer_output = self.trm_layer(trm_input_feature) 
        transformer_output = torch.flatten(transformer_output,start_dim=1) 
        mlp_input_feature = torch.cat((other_embedding, transformer_output), dim=1) 
        bst_out = self.mlp(mlp_input_feature) 
        if self.use_wide:
            lr_feature_map = self.lr_sparse(x)
            lr_out = self.lr(lr_feature_map)
            bst_out += lr_out
        if self.use_deep:
            nn_feature_map = self.deep_sparse(x)
            deep_out = self.deep(nn_feature_map)
            bst_out += deep_out
        return self.final_activation(bst_out)
