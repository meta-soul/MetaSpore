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
from ...layers import MLPLayer, InterestExtractorNetwork, InterestEvolvingLayer, SequenceAttLayer, LRLayer
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pack_sequence, pad_sequence
import numpy as np


class DIEN(torch.nn.Module):
    def __init__(self,
            column_name_path = None,
            dien_combine_schema_path = None,
            wide_combine_schema_path = None,
            deep_combine_schema_path = None,
            sparse_init_var = 0.01,
            use_wide = True,
            use_deep = True,
                 
            post_item_seq = [1],
            neg_item_seq = [2],
            target_item = [3],
                 
            dien_embedding_size = 30,
            dien_gru_num_layer = 1,
                 
            dien_aux_hidden_units = [32,16],
            dien_use_aux_bn = False,
            dien_aux_dropout = 0,
            dien_aux_activation = 'Sigmoid',

            dien_att_hidden_units = [40],
            dien_use_att_bn = False,
            dien_att_dropout = 0,
            dien_att_activation = 'Sigmoid',

            dien_dnn_hidden_units = [64,16],
            dien_use_dnn_bn = True,
            dien_dnn_dropout = 0.1,
            dien_dnn_activation = 'Dice',

            target_loss_weight = 1.0,
            auxilary_loss_weight = 1.0,

            deep_hidden_units = [64,16],
            deep_dropout = 0.1,
            deep_activation = 'relu',
            use_deep_bn = True,
            use_deep_bias = True,
            deep_embedding_size = 30,

            wide_embedding_size = 30,
            ):
        super().__init__()
        self.use_wide = use_wide
        self.use_deep = use_deep
        self.post_item_seq_index = post_item_seq
        self.neg_item_seq_index = neg_item_seq
        self.target_item_index = target_item
        
        # wide
        if self.use_wide:
            self.lr_sparse = ms.EmbeddingSumConcat(wide_embedding_size,
                                                   column_name_path,
                                                   wide_combine_schema_path)
            self.lr_sparse.updater = ms.FTRLTensorUpdater()
            self.lr_sparse.initializer = ms.NormalTensorInitializer(var=sparse_init_var)
            self.lr = LRLayer(self.embedding_size, self.lr_sparse.feature_count)
  
        # deep
        if self.use_deep:
            self.dnn_sparse = ms.EmbeddingSumConcat(deep_embedding_size,
                                                    column_name_path,
                                                    deep_combine_schema_path)
            self.dnn_sparse.updater = ms.FTRLTensorUpdater()
            self.dnn_sparse.initializer = ms.NormalTensorInitializer(var=sparse_init_var)
            self.dnn = MLPLayer(input_dim = self.dnn_sparse.feature_count * deep_embedding_size ,
                                output_dim = 1,
                                hidden_units = deep_hidden_units,
                                hidden_activations = deep_activation,
                                final_activation = deep_activation, 
                                dropout_rates = deep_dropout, 
                                batch_norm = use_deep_bn, 
                                use_bias = use_deep_bias)
            
        # dien
        self.dien_embedding_layer = ms.EmbeddingLookup(dien_embedding_size, 
                                                       column_name_path, 
                                                       dien_combine_schema_path)
        self.dien_embedding_layer.updater = ms.FTRLTensorUpdater()
        self.dien_embedding_layer.itializer = ms.NormalTensorInitializer(var=sparse_init_var)
        
         # feature count
        self.total_feature_count = self.dien_embedding_layer.feature_count
        self.item_feature_count = len(target_item)
        self.non_seq_feature_count = self.total_feature_count - 3*self.item_feature_count
        
        # self.aux_units = [2*dien_embedding_size*self.item_feature_count] + dien_aux_hidden_units
        # self.att_units = [4*self.embedding_size*self.item_feature_number] + att_hidden_units
        # self.dnn_units = [(2*self.item_feature_number+self.user_feature_number)*self.embedding_size] + dnn_hidden_units

        self.intereset_extractor = InterestExtractorNetwork(self.embedding_size, 
                                                            self.aux_units, 
                                                            self.embedding_size, 
                                                            self.gru_num_layer, 
                                                            aux_activation, 
                                                            aux_dropout, 
                                                            use_aux_bn)
        
        self.interest_evolution = InterestEvolvingLayer(self.embedding_size, 
                                                        self.embedding_size, 
                                                        self.gru_num_layer, 
                                                        self.att_units, 
                                                        att_activation, 
                                                        att_dropout, 
                                                        use_att_bn, 
                                                        max_length)
        
        self.dnn_predict_layers = MLPLayer(input_dim = (2*self.item_feature_count+self.non_seq_feature_count)*dien_embedding_size, 
                                       hidden_units = dien_dnn_hidden_units, 
                                       output_dim = 1,
                                       hidden_activations = dien_dnn_activation, 
                                       final_activation = dien_dnn_activation,
                                       dropout_rates = dien_dnn_dropout, 
                                       batch_norm = dien_use_dnn_bn)
        
        self.dnn_predict_layer = torch.nn.Linear(self.dnn_units[-1], 1)
        
    def get_field_embedding_list(x):
        x, offset = self.dien_embedding_layer(x) 
        # reshape
        x_reshape = [x[offset[i]:offset[i+1],:] for i in range(offset.shape[0]-1)]
        x_reshape.append(x[offset[offset.shape[0]-1]:x.shape[0],:])
        return x_reshape

    def forward(self, x):
        x = self.get_field_embedding_list(x) 
        
        # calculate sequence length
        seq_length = [item.shape[0] for item in x_reshape[self.feature_slice['item_seq'][0]::self.total_feature]]
        
        # split feature
        post_item_seq, neg_item_seq, target_item, non_seq_feature = [], [], [], []
        for feature_index in range(self.total_feature_count):
        # for every column
            if feature_index in self.post_item_seq_index: # collect positive item sequence feature
                post_item_seq.append(torch.nn.utils.rnn.pad_sequence(x_reshape[feature_index::self.total_feature_count],batch_first=True))
                
            if feature_index in self.neg_item_seq_index: # collect negtive item sequence feature
                neg_item_seq.append(torch.nn.utils.rnn.pad_sequence(x_reshape[feature_index::self.total_feature_count],batch_first=True))
                
            if feature_index in self.target_item_index: # collect target item feature
                target_item.append(torch.stack(x_reshape[feature_index::self.total_feature_count]).squeeze())
                
            else:# collect non sequence feature
                non_seq_feature.append(torch.stack(x_reshape[feature_index::self.total_feature_count]).squeeze())
         
        # concat four categories of feature
        target_item, non_seq_feature = torch.cat(target_item, dim=-1), torch.cat(non_seq_feature, dim=-1)
        post_item_seq_pack = torch.nn.utils.rnn.pack_padded_sequence(torch.cat(post_item_seq, dim=-1), seq_length , batch_first=True, enforce_sorted=False)
        neg_item_seq_pack = torch.nn.utils.rnn.pack_padded_sequence(torch.cat(neg_item_seq, dim=-1), seq_length , batch_first=True, enforce_sorted=False)
        
#         # split feature
#         feature = {}
#         total_feature = np.sum([len(index) for index in self.feature_slice.values()])
#         seq_length = [item.shape[0] for item in x_reshape[self.feature_slice['item_seq'][0]::self.total_feature]]
#         for keys in self.feature_slice.keys():
#             keys_list = []
#             for slice_index in self.feature_slice[keys]:
#                 single_feature = torch.squeeze(pad_sequence(x_reshape[slice_index::self.total_feature],batch_first=True))
#                 keys_list.append(single_feature)
#             feature[keys] = torch.cat(keys_list, dim=-1)

#         user = feature['user']
#         target_item = feature['target_item']
#         item_seq_pack = pack_padded_sequence(feature['item_seq'], seq_length , batch_first=True, enforce_sorted=False)
#         neg_item_seq_pack = pack_padded_sequence(feature['neg_item_seq'], seq_length , batch_first=True, enforce_sorted=False)

        
        interest, aux_loss = self.intereset_extractor(post_item_seq_pack, neg_item_seq_pack)
        evolution = self.interest_evolution(target_item, interest)
        dien_in = torch.cat([ target_item, non_seq_feature, evolution], dim=-1)
        preds = self.self.dnn_predict_layers(dien_in)

        if self.use_wide:
            lr_feature_map = self.lr_sparse(x)
            lr_logit = self.lr(lr_feature_map)
            preds += lr_logit
        if self.use_deep:
            nn_feature_map = self.dnn_sparse(x)
            dnn_logit = self.dnn(nn_feature_map)
            preds += dnn_logit
            
        return torch.sigmoid(preds), aux_loss
