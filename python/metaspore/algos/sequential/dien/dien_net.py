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
from ...layers import MLPLayer, InterestExtractorNetwork, InterestEvolvingLayer, LRLayer
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
            pos_item_seq = [1],
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
            dien_use_gru_bias = False,
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

        # wide
        if self.use_wide:
            self.lr_sparse = ms.EmbeddingSumConcat(wide_embedding_size,
                                                   column_name_path,
                                                   wide_combine_schema_path)
            self.lr_sparse.updater = ms.FTRLTensorUpdater()
            self.lr_sparse.initializer = ms.NormalTensorInitializer(var=sparse_init_var)
            self.lr = LRLayer(wide_embedding_size, self.lr_sparse.feature_count)

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

        # The number of columns assigned to target_item,pos_item_seq and neg_item_seq must be same
        assert len(target_item) == len(pos_item_seq) and len(target_item) == len(neg_item_seq) and len(pos_item_seq) == len(neg_item_seq),"The number of columns assigned to target_item,pos_item_seq and neg_item_seq must be same"


        # feature count
        self.total_feature_count = self.dien_embedding_layer.feature_count
        self.item_feature_count = len(target_item)
        self.non_seq_feature_count = self.total_feature_count - 3*self.item_feature_count


        self.pos_item_seq_index = pos_item_seq
        self.neg_item_seq_index = neg_item_seq
        self.target_item_index = target_item
        self.non_seq_index = [i for i in range(self.total_feature_count) if i not in self.pos_item_seq_index+self.neg_item_seq_index+self.target_item_index]


        self.intereset_extractor = InterestExtractorNetwork(embedding_size = dien_embedding_size*self.item_feature_count,
                                                            gru_hidden_size = dien_embedding_size*self.item_feature_count,
                                                            gru_num_layers = dien_gru_num_layer,
                                                            use_gru_bias = dien_use_gru_bias,
                                                            aux_input_dim = 2*dien_embedding_size*self.item_feature_count,
                                                            aux_hidden_units = dien_aux_hidden_units,
                                                            aux_activation = dien_aux_activation,
                                                            aux_dropout =  dien_aux_dropout,
                                                            use_aux_bn = dien_use_aux_bn)

        self.interest_evolution = InterestEvolvingLayer(embedding_size = dien_embedding_size*self.item_feature_count,
                                                        gru_hidden_size = dien_embedding_size*self.item_feature_count,
                                                        gru_num_layers = dien_gru_num_layer,
                                                        use_gru_bias = dien_use_gru_bias,
                                                        att_input_dim = dien_embedding_size*self.item_feature_count,
                                                        att_hidden_units = dien_att_hidden_units,
                                                        att_activation = dien_att_activation,
                                                        att_dropout = dien_att_dropout,
                                                        use_att_bn = dien_use_att_bn,
                                                        )


        self.dnn_predict_layers = MLPLayer(input_dim = (2*self.item_feature_count+self.non_seq_feature_count)*dien_embedding_size,
                                       hidden_units = dien_dnn_hidden_units,
                                       output_dim = 1,
                                       hidden_activations = dien_dnn_activation,
                                       final_activation = dien_dnn_activation,
                                       dropout_rates = dien_dnn_dropout,
                                       batch_norm = dien_use_dnn_bn)


    def get_field_embedding_list(self, x):
        x, offset = self.dien_embedding_layer(x)
        # reshape
        x_reshape = [x[offset[i]:offset[i+1],:] for i in range(offset.shape[0]-1)]
        x_reshape.append(x[offset[offset.shape[0]-1]:x.shape[0],:])
        return x_reshape

    def get_feature_concat(self,x,slice_index_list):
        feature_list = []
        for slice_index in slice_index_list:
            single_feature = torch.squeeze(pad_sequence(x[slice_index::self.total_feature_count],batch_first=True))
            feature_list.append(single_feature)
        return torch.cat(feature_list, dim=-1)

    def forward(self, x):
        x = self.get_field_embedding_list(x)

        # calculate sequence length
        seq_length = [seq.shape[0] for seq in x[self.pos_item_seq_index[0]::self.total_feature_count]]

        # split feature
        non_seq_feature = self.get_feature_concat(x, self.non_seq_index)
        target_item = self.get_feature_concat(x, self.target_item_index)
        pos_item_seq = self.get_feature_concat(x, self.pos_item_seq_index)
        neg_item_seq = self.get_feature_concat(x, self.neg_item_seq_index)

        # pack sequential feature
        pos_item_seq_pack = torch.nn.utils.rnn.pack_padded_sequence(input = pos_item_seq,
                                                                     lengths = seq_length ,
                                                                     batch_first = True,
                                                                     enforce_sorted = False)
        neg_item_seq_pack = torch.nn.utils.rnn.pack_padded_sequence(input = neg_item_seq,
                                                                    lengths = seq_length ,
                                                                    batch_first = True,
                                                                    enforce_sorted = False)

        interest, aux_loss = self.intereset_extractor(pos_item_seq_pack, neg_item_seq_pack)
        evolution = self.interest_evolution(target_item, interest)
        dien_in = torch.cat([ target_item, non_seq_feature, evolution], dim=-1)
        logit = self.dnn_predict_layers(dien_in)

        if self.use_wide:
            lr_feature_map = self.lr_sparse(x)
            lr_logit = self.lr(lr_feature_map)
            logit += lr_logit
        if self.use_deep:
            nn_feature_map = self.dnn_sparse(x)
            dnn_logit = self.dnn(nn_feature_map)
            logit += dnn_logit

        return torch.sigmoid(logit), aux_loss
