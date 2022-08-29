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
            embedding_size = 30,
            aux_hidden_units = [32,16],
            use_aux_bn = False,
            aux_dropout = 0,
            aux_activation = 'Sigmoid',
            gru_num_layer = 1,
            att_hidden_units = [40],
            use_att_bn = False,
            att_dropout = 0,
            att_activation = 'Sigmoid',
            dnn_hidden_units = [64,16],
            use_dnn_bn = True,
            dnn_dropout = 0.1,
            dnn_activation = 'Dice', 
            column_name_path = None,
            combine_schema_path = None,
            sparse_init_var = 0.01,
            feature_slice = None,
            use_wide = False,
            use_deep = False,
            wide_deep_combine_schema_path = None,
            deep_hidden_units = [64,16],
            deep_dropout = 0.1,
            deep_activation = 'relu',
            use_deep_bn = True,
            use_deep_bias = True,
            max_length = 10,
            ):
        super().__init__()
        self.use_wide = use_wide
        self.use_deep = use_deep
        self.feature_slice = feature_slice
        self.embedding_size = embedding_size
        self.total_feature = np.sum([len(index) for index in self.feature_slice.values()])
        self.user_feature_number = len(feature_slice['user'])
        self.item_feature_number = len(feature_slice['target_item'])
        self.aux_units = [2*self.embedding_size*self.item_feature_number] + aux_hidden_units
        self.gru_num_layer = gru_num_layer
        self.att_units = [4*self.embedding_size*self.item_feature_number] + att_hidden_units
        self.dnn_units = [(2*self.item_feature_number+self.user_feature_number)*self.embedding_size] + dnn_hidden_units
        self.column_name_path = column_name_path
        self.combine_schema_path = combine_schema_path
        self.wide_deep_combine_schema_path = wide_deep_combine_schema_path
        
        if self.use_wide:
            self.lr_sparse = ms.EmbeddingSumConcat(self.embedding_size,
                                                   self.column_name_path,
                                                   self.wide_deep_combine_schema_path)
            self.lr_sparse.updater = ms.FTRLTensorUpdater()
            self.lr_sparse.initializer = ms.NormalTensorInitializer(var=sparse_init_var)
            self.lr = LRLayer(self.embedding_size, self.lr_sparse.feature_count)
  
        
        if self.use_deep:
            self.dnn_sparse = ms.EmbeddingSumConcat(self.embedding_size,
                                                self.column_name_path,
                                                self.wide_deep_combine_schema_path)
            self.dnn_sparse.updater = ms.FTRLTensorUpdater()
            self.dnn_sparse.initializer = ms.NormalTensorInitializer(var=sparse_init_var)
            self.dnn = MLPLayer(input_dim = self.dnn_sparse.feature_count * self.embedding_size,
                                output_dim = 1,
                                hidden_units = deep_hidden_units,
                                hidden_activations = deep_activation,
                                final_activation = deep_activation, 
                                dropout_rates = deep_dropout, 
                                batch_norm = use_deep_bn, 
                                use_bias = use_deep_bias)

        self.embedding_layer = ms.EmbeddingLookup(self.embedding_size, self.column_name_path, self.combine_schema_path)
        self.embedding_layer.updater = ms.FTRLTensorUpdater()
        self.embedding_layer.itializer = ms.NormalTensorInitializer(var=sparse_init_var)

        self.intereset_extractor = InterestExtractorNetwork(self.embedding_size, self.aux_units, self.embedding_size, self.gru_num_layer, aux_activation, aux_dropout, use_aux_bn)
        self.interest_evolution = InterestEvolvingLayer(self.embedding_size, self.embedding_size, self.gru_num_layer, self.att_units, att_activation, att_dropout, use_att_bn, max_length)

        self.dnn_mlp_layers = MLPLayer(input_dim=self.dnn_units[0], hidden_units=self.dnn_units[1:], hidden_activations=dnn_activation, dropout_rates=dnn_dropout, batch_norm=use_dnn_bn)
        self.dnn_predict_layer = torch.nn.Linear(self.dnn_units[-1], 1)

    def forward(self, x):
        x, offset = self.embedding_layer(x) 
        # reshape
        x_reshape = [x[offset[i]:offset[i+1],:] for i in range(offset.shape[0]-1)]
        x_reshape.append(x[offset[offset.shape[0]-1]:x.shape[0],:])
        
        # split feature
        feature = {}
        total_feature = np.sum([len(index) for index in self.feature_slice.values()])
        seq_length = [item.shape[0] for item in x_reshape[self.feature_slice['item_seq'][0]::self.total_feature]]
        for keys in self.feature_slice.keys():
            keys_list = []
            for slice_index in self.feature_slice[keys]:
                single_feature = torch.squeeze(pad_sequence(x_reshape[slice_index::self.total_feature],batch_first=True))
                keys_list.append(single_feature)
            feature[keys] = torch.cat(keys_list, dim=-1)

        user = feature['user']
        target_item = feature['target_item']
        item_seq_pack = pack_padded_sequence(feature['item_seq'], seq_length , batch_first=True, enforce_sorted=False)
        neg_item_seq_pack = pack_padded_sequence(feature['neg_item_seq'], seq_length , batch_first=True, enforce_sorted=False)

        
        
        interest, aux_loss = self.intereset_extractor(item_seq_pack, neg_item_seq_pack)
        evolution = self.interest_evolution(target_item, interest)
        dien_in = torch.cat([ target_item, user, evolution], dim=-1)
        dien_out = self.dnn_mlp_layers(dien_in)
        preds = self.dnn_predict_layer(dien_out)

        if self.use_wide:
            lr_feature_map = self.lr_sparse(x)
            lr_logit = self.lr(lr_feature_map)
            preds += lr_logit
        if self.use_deep:
            nn_feature_map = self.dnn_sparse(x)
            dnn_logit = self.dnn(nn_feature_map)
            preds += dnn_logit
            
        return torch.sigmoid(preds), aux_loss
