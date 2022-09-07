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
import torch.nn.functional as F

class GRU4RecSimilarityModule(torch.nn.Module):
    def __init__(self, tau=1.0, net_dropout=0.0):
        super().__init__()
        self.tau = tau
        self.user_dropout = torch.nn.Dropout(net_dropout) if net_dropout > 0 else None
    
    def forward(self, x, y):
        z = torch.sum(x * y, dim=1).reshape(-1, 1)
        return torch.sigmoid(z/self.tau)

class GRU4RecUserModule(torch.nn.Module):
    def __init__(self,
                 column_name_path, 
                 seq_combine_schema_path,
                 gru4rec_embedding_dim,
                 gru_hidden_dim,
                 gru_num_layers,
                 net_dropout=0,
                 embedding_regularizer=None,
                 net_regularizer=None,
                 sparse_init_var=1e-4,
                 ftrl_l1=1.0,
                 ftrl_l2=120.0,
                 ftrl_alpha=0.5,
                 ftrl_beta=1.0):
        super().__init__()
        # pooling type checks
        self.embedding_table = ms.EmbeddingLookup(gru4rec_embedding_dim, column_name_path, seq_combine_schema_path)
        self.embedding_table.updater = ms.FTRLTensorUpdater(l1=ftrl_l1, l2=ftrl_l2, alpha = ftrl_alpha, beta=ftrl_beta)
        self.embedding_table.initializer = ms.NormalTensorInitializer(var=sparse_init_var)

        self.embedding_table.output_batchsize1_if_only_level0 = True

        self.gru_layers = torch.nn.GRU(input_size=gru4rec_embedding_dim,
                                        hidden_size=gru_hidden_dim,
                                        num_layers=gru_num_layers,
                                        bias=False,
                                        batch_first=True,
                                        )
        # dropout
        self.embedding_dropout = torch.nn.Dropout(net_dropout) if net_dropout > 0 else None
    
    def get_field_embedding_list(self, x, offset):
        x_reshape = [x[offset[i]:offset[i+1],:] for i in range(offset.shape[0]-1)]
        x_reshape.append(x[offset[offset.shape[0]-1]:x.shape[0],:])
        return x_reshape 

    def gather_indexes(self, output, gather_index):
        """Gathers the vectors at the specific positions over a minibatch"""
        # output: [B, max_len, H]
        # gather_index: [B]
        gather_index = gather_index.view(-1, 1, 1).expand(-1, -1, output.shape[-1])
        output_tensor = output.gather(dim=1, index=gather_index)
        return output_tensor.squeeze(1)
    
    def forward(self, x):
        x, offset = self.embedding_table(x) 
        # print("x.shape: ", x.shape)
        # print("x: ", x)
        # print(offset)
        x_reshape = self.get_field_embedding_list(x, offset)
        item_seq_emb = torch.nn.utils.rnn.pad_sequence(x_reshape, batch_first=True)
        start_idx = 1
        item_seq_length = [offset[i] - offset[i-1] for i in range(start_idx, offset.shape[0])]
        item_seq_length.append(x.shape[0]-offset[-1])
        item_seq_length = torch.tensor(item_seq_length)
        if self.embedding_dropout:
            item_seq_emb_dropout = self.emb_dropout(item_seq_emb)
        gru_output, _ = self.gru_layers(item_seq_emb_dropout)
        gru_output = self.dense(gru_output)
        seq_output = self.gather_indexes(gru_output, item_seq_length - 1) 
        return F.normalize(seq_output)
    
    
class GRU4RecItemModule(torch.nn.Module):
    def __init__(self, 
                 column_name_path, 
                 combine_schema_path, 
                 embedding_dim,
                 pooling_type_layer_1='sum', # in ['mean', 'sum', 'max'] 
                 net_dropout=0,
                 embedding_regularizer=None,
                 net_regularizer=None,
                 sparse_init_var=1e-4,
                 ftrl_l1=1.0,
                 ftrl_l2=120.0,
                 ftrl_alpha=0.5,
                 ftrl_beta=1.0):
        super().__init__()
        # pooling type checks
        if pooling_type_layer_1 not in ('mean', 'sum', 'max'):
            raise ValueError(f"Pooling type must be one of: 'mean', 'sum', 'max'; {pooling_type_layer_1!r} is invalid")
         # user embedding and pooling layer1
        self.sparse = ms.EmbeddingSumConcat(
            embedding_dim, 
            column_name_path, 
            combine_schema_path,
            embedding_bag_mode=pooling_type_layer_1
        )
        self.sparse.updater = ms.FTRLTensorUpdater(
            l1=ftrl_l1,
            l2=ftrl_l2, 
            alpha=ftrl_alpha, 
            beta=ftrl_beta
        )        
        self.sparse.initializer = ms.NormalTensorInitializer(var = sparse_init_var)
        # dropout
        self.embedding_dropout = torch.nn.Dropout(net_dropout) if net_dropout > 0 else None

    def forward(self, x):
        x = self.sparse(x)
        if self.embedding_dropout:
            x = self.embedding_dropout(x)
        x = F.normalize(x)
        return x
