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
import torch.nn.functional as F
import metaspore as ms
import math, copy
from torch.nn.utils.rnn import pack_sequence, pack_padded_sequence, pad_packed_sequence

# Logistic regression layer
class LRLayer(torch.nn.Module):
    def __init__(self,
                embedding_size,
                feature_dim):
        super().__init__()
        self.embedding_size = embedding_size
        self.feature_dim = feature_dim

    def forward(self, inputs):
        out = torch.sum(inputs, dim=1, keepdim=True)
        return out

# This code is adapted from github repository: https://github.com/RUCAIBox/RecBole/blob/master/recbole/model/layers.py
class Dice(torch.nn.Module):

    def __init__(self, hidden_size):
        super(Dice, self).__init__()

        self.sigmoid = torch.nn.Sigmoid()
        self.alpha = torch.nn.Parameter(torch.zeros((hidden_size,)))

    def forward(self, score):
        self.alpha = self.alpha.to(score.device)
        score_p = self.sigmoid(score)

        return self.alpha * (1 - score_p) * score + score_p * score


# Fully connected layers
# This code is adapted from github repository:  https://github.com/xue-pai/FuxiCTR/blob/main/fuxictr/pytorch/layers/deep.py
class MLPLayer(torch.nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim=None,
                 hidden_units=[],
                 hidden_activations="ReLU",
                 final_activation=None,
                 dropout_rates=0,
                 batch_norm=False,
                 use_bias=True,
                 input_norm=False):
        super().__init__()
        dense_layers = []
        if not isinstance(dropout_rates, list):
            dropout_rates = [dropout_rates] * len(hidden_units)
        if not isinstance(hidden_activations, list):
            hidden_activations = [hidden_activations] * len(hidden_units)
        hidden_activations = [self.set_activation(x, hidden_size) for x, hidden_size in zip(hidden_activations, hidden_units)]
        hidden_units = [input_dim] + hidden_units

        if input_norm:
            dense_layers.append(ms.nn.Normalization(input_dim))

        ## batchnorm + linear + activation + dropout...
        for idx in range(len(hidden_units) - 1):
            if batch_norm:
                dense_layers.append(ms.nn.Normalization(hidden_units[idx]))
            dense_layers.append(torch.nn.Linear(hidden_units[idx], hidden_units[idx + 1], bias = use_bias))
            if hidden_activations[idx] is not None \
                and (idx < len(hidden_units) - 2 or output_dim is not None):
                dense_layers.append(hidden_activations[idx])
            if dropout_rates[idx] is not None and dropout_rates[idx] > 0 \
                and (idx < len(hidden_units) - 2 or output_dim is not None):
                dense_layers.append(torch.nn.Dropout(p=dropout_rates[idx]))
        ## final layer
        if output_dim is not None:
            dense_layers.append(torch.nn.Linear(hidden_units[-1], output_dim, bias=use_bias))
        ## final activation
        if final_activation is not None:
            dense_layers.append(self.set_activation(final_activation, output_dim))
        ## all in one
        self.dnn = torch.nn.Sequential(*dense_layers)

    def forward(self, inputs):
        return self.dnn(inputs)

    @staticmethod
    def set_activation(activation, hidden_size):
        if isinstance(activation, str):
            if activation.lower() == "relu":
                return torch.nn.ReLU()
            elif activation.lower() == "sigmoid":
                return torch.nn.Sigmoid()
            elif activation.lower() == "tanh":
                return torch.nn.Tanh()
            elif activation.lower() == "dice":
                return Dice(hidden_size)
            else:
                return torch.nn.ReLU() ## defalut relu
        else:
            return torch.nn.ReLU() ## defalut relu

# Factorization machine layer
class FMLayer(torch.nn.Module):
    def __init__(self,
                feature_count,
                embedding_dim):
        super().__init__()
        self._feature_count = feature_count
        self._embedding_dim = embedding_dim

    def forward(self, inputs):
        inputs = inputs.reshape(-1, self._feature_count, self._embedding_dim)
        square_of_sum = torch.pow(torch.sum(inputs, dim=1, keepdim=True), 2)
        sum_of_square = torch.sum(inputs * inputs, dim=1, keepdim=True)
        cross_term = square_of_sum - sum_of_square
        cross_term = 0.5 * torch.sum(cross_term, dim=2, keepdim=False)
        return cross_term


# Field-aware Factorization machine layer
class FFMLayer(torch.nn.Module):
    def __init__(self,
                 feature_count,
                 embedding_dim,
                 embedding_module_list):
        super().__init__()
        self._feature_count = feature_count
        self._embedding_dim = embedding_dim
        self._embedding_module_list = embedding_module_list

    def forward(self, inputs):
        # [B, F x E] = > [[B, F x E]] => [[B, F, E]]
        field_aware_embedding_list = [
            each_layer(inputs).reshape(-1, self._feature_count, self._embedding_dim)
            for each_layer in self._embedding_module_list
        ]

        dot_sum = 0
        for i in range(self._feature_count - 1):
            for j in range(i + 1, self._feature_count):
                embedding_ij = field_aware_embedding_list[j - 1][:, i, :]
                embedding_ji = field_aware_embedding_list[i][:, j, :]
                dot_sum += torch.sum(embedding_ij * embedding_ji, dim=1, keepdim=True)
        return dot_sum

# Cross net layer
# This code is adapted from github repository:  https://github.com/xue-pai/FuxiCTR/blob/main/fuxictr/pytorch/layers/interaction.py
class CrossNet(torch.nn.Module):
    def __init__(self, input_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.cross_net = torch.nn.ModuleList(CrossInteractionLayer(input_dim)
                                       for _ in range(self.num_layers))

    def forward(self, X_0):
        X_i = X_0
        for i in range(self.num_layers):
            X_i = X_i + self.cross_net[i](X_0,X_i)
        return X_i

# Cross interaction Layer
# This code is adapted from github repository:  https://github.com/xue-pai/FuxiCTR/blob/main/fuxictr/pytorch/layers/interaction.py
class CrossInteractionLayer(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.weight = torch.nn.Linear(input_dim, 1, bias=False)
        self.bias = torch.nn.Parameter(torch.zeros(input_dim))

    def forward(self, X_0, X_i):
        interaction_out = self.weight(X_i) * X_0 + self.bias
        return interaction_out


# Multi-head attention Layer
# This code is adapted from github repository:  https://github.com/xue-pai/FuxiCTR/blob/main/fuxictr/pytorch/layers/attention.py
class MultiHeadAttention(torch.nn.Module):
    def __init__(self, input_dim, dim_per_head=None, num_heads=1, dropout_rate=None,
                 use_residual=True, layer_norm=False):
        super().__init__()
        self.dim_per_head = dim_per_head
        self.num_heads = num_heads
        self.output_dim = self.dim_per_head * self.num_heads
        self.use_residual = use_residual
        self.W_query = torch.nn.Linear(input_dim, self.output_dim, bias=False)
        self.W_key = torch.nn.Linear(input_dim, self.output_dim, bias=False)
        self.W_value = torch.nn.Linear(input_dim, self.output_dim, bias=False)
        self.W_residual = torch.nn.Linear(input_dim, self.output_dim, bias=False)
        self.dot_product_attention = ScaledDotproductAttention(dropout_rate)
        self.layer_norm = torch.nn.LayerNorm(self.output_dim) if layer_norm else None
        self.dropout = torch.nn.Dropout(dropout_rate) if dropout_rate is not None and dropout_rate > 0 else None

    def forward(self, query, key, value):
        ## linear projection
        residual = self.W_residual(query)
        query = self.W_query(query)
        key = self.W_key(key)
        value = self.W_value(value)
        ## split by heads
        batch_size = query.size(0)
        query = query.view(batch_size * self.num_heads, -1, self.dim_per_head)
        key = key.view(batch_size * self.num_heads, -1, self.dim_per_head)
        value = value.view(batch_size * self.num_heads, -1, self.dim_per_head)
        ## scaled dot product attention
        output, attention = self.dot_product_attention(query, key, value)
        ## concat heads
        output = output.view(batch_size, -1, self.output_dim)
        ## dropout
        if self.dropout is not None:
            output = self.dropout(output)
        ## use residual
        if self.use_residual:
            residual = residual.view(batch_size, -1, self.output_dim)
            output = output + residual
        ## use layernorm
        if self.layer_norm is not None:
            output = self.layer_norm(output)
        ## activation
        output = output.relu()
        ## result
        return output, attention

# Multi-head self attention Layer
# This code is adapted from github repository:  https://github.com/xue-pai/FuxiCTR/blob/main/fuxictr/pytorch/layers/attention.py
class MultiHeadSelfAttention(MultiHeadAttention):
    def forward(self,X):
        output, attention = super().forward(X, X, X)
        return output

# Scaled dot-product attention layer
# Copyright (C) 2018. pengshuang@Github for ScaledDotProductAttention.
class ScaledDotproductAttention(torch.nn.Module):
    def __init__(self, dropout_rate=0.):
        super().__init__()
        self.dropout = None
        if dropout_rate is not None and dropout_rate > 0:
            self.dropout = torch.nn.Dropout(dropout_rate)
        self.softmax = torch.nn.Softmax(dim=2)

    def forward(self, W_q, W_k, W_v, scale=None, mask=False):
        attention=torch.bmm(W_q, W_k.transpose(1,2))
        scale = (W_k.size(-1)) ** -0.5
        attention = attention * scale
        if mask:
            mask = torch.triu(torch.ones(attention.size(1), attention.size(2)), 1).bool()
            attention.masked_fill_(mask, 0)
        attention = self.softmax(attention)
        if self.dropout is not None:
            attention = self.dropout(attention)
        output = torch.bmm(attention, W_v)
        return output, attention

# Cross net v2 layer
# This code is adapted from github repository:  https://github.com/xue-pai/FuxiCTR/blob/main/fuxictr/pytorch/models/DCNv2.py
class CrossNetV2(torch.nn.Module):
    def __init__(self, input_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.cross_layers = torch.nn.ModuleList(torch.nn.Linear(input_dim, input_dim)
                                          for _ in range(self.num_layers))

    def forward(self, X_0):
        X_i = X_0 # b x dim
        for i in range(self.num_layers):
            X_i = X_i + X_0 * self.cross_layers[i](X_i)
        return X_i


# Cross net mix Layer
# Copyright (C) 2021 The DeepCTR-Torch authors for CrossNetMix
class CrossNetMix(torch.nn.Module):
    """ CrossNetMix improves CrossNet by:
        1. add MOE to learn feature interactions in different subspaces
        2. add nonlinear transformations in low-dimensional space
    """

    def __init__(self, in_features, layer_num=2, low_rank=32, num_experts=4):
        super().__init__()
        self.layer_num = layer_num
        self.num_experts = num_experts

        # U: (in_features, low_rank)
        self.U_list = torch.nn.ParameterList([torch.nn.Parameter(torch.nn.init.xavier_normal_(
            torch.empty(num_experts, in_features, low_rank))) for i in range(self.layer_num)])
        # V: (in_features, low_rank)
        self.V_list = torch.nn.ParameterList([torch.nn.Parameter(torch.nn.init.xavier_normal_(
            torch.empty(num_experts, in_features, low_rank))) for i in range(self.layer_num)])
        # C: (low_rank, low_rank)
        self.C_list = torch.nn.ParameterList([torch.nn.Parameter(torch.nn.init.xavier_normal_(
            torch.empty(num_experts, low_rank, low_rank))) for i in range(self.layer_num)])
        self.gating = torch.nn.ModuleList([torch.nn.Linear(in_features, 1, bias=False) for i in range(self.num_experts)])

        self.bias = torch.nn.ParameterList([torch.nn.Parameter(torch.nn.init.zeros_(
            torch.empty(in_features, 1))) for i in range(self.layer_num)])

    def forward(self, inputs):
        x_0 = inputs.unsqueeze(2)  # (bs, in_features, 1)
        x_l = x_0
        for i in range(self.layer_num):
            output_of_experts = []
            gating_score_of_experts = []
            for expert_id in range(self.num_experts):
                # (1) G(x_l)
                # compute the gating score by x_l
                gating_score_of_experts.append(self.gating[expert_id](x_l.squeeze(2)))

                # (2) E(x_l)
                # project the input x_l to $\mathbb{R}^{r}$
                v_x = torch.matmul(self.V_list[i][expert_id].t(), x_l)  # (bs, low_rank, 1)

                # nonlinear activation in low rank space
                v_x = torch.tanh(v_x)
                v_x = torch.matmul(self.C_list[i][expert_id], v_x)
                v_x = torch.tanh(v_x)

                # project back to $\mathbb{R}^{d}$
                uv_x = torch.matmul(self.U_list[i][expert_id], v_x)  # (bs, in_features, 1)

                dot_ = uv_x + self.bias[i]
                dot_ = x_0 * dot_  # Hadamard-product

                output_of_experts.append(dot_.squeeze(2))

            # (3) mixture of low-rank experts
            output_of_experts = torch.stack(output_of_experts, 2)  # (bs, in_features, num_experts)
            gating_score_of_experts = torch.stack(gating_score_of_experts, 1)  # (bs, num_experts, 1)
            moe_out = torch.matmul(output_of_experts, gating_score_of_experts.softmax(1))
            x_l = moe_out + x_l  # (bs, in_features, 1)

        x_l = x_l.squeeze()  # (bs, in_features)
        return x_l


# Inner product layer
class InnerProductLayer(torch.nn.Module):
    def __init__(self, num_fields=None, embedding_dim=None):
        super().__init__()
        self.num_fields = num_fields
        self.embedding_dim = embedding_dim
        self.interaction_units = int(num_fields * (num_fields - 1) / 2)
        self.upper_triange_mask = torch.triu(torch.ones(num_fields, num_fields), 1).byte()

    def forward(self, inputs):
        inputs = inputs.view(-1, self.num_fields, self.embedding_dim)
        inner_product_matrix = torch.bmm(inputs, inputs.transpose(1, 2))
        flat_upper_triange = torch.masked_select(inner_product_matrix, self.upper_triange_mask)
        return flat_upper_triange.view(-1, self.interaction_units)


# Outer product layer
class OuterProductLayer(torch.nn.Module):
    def __init__(self, num_fields=None, embedding_dim=None):
        super().__init__()
        self.num_fields = num_fields
        self.embedding_dim = embedding_dim
        self.interaction_units = int(embedding_dim * (embedding_dim - 1) / 2)
        self.upper_triange_mask = torch.triu(torch.ones(self.embedding_dim, self.embedding_dim), 1).byte()

    def forward(self, inputs):
        inputs = inputs.view(-1, self.num_fields, self.embedding_dim)
        inputs = torch.transpose(inputs, 1 , 2)
        hadamard_tensor = torch.einsum("bhd,bmd->bhmd", inputs, inputs)
        hadamard_tensor = hadamard_tensor.contiguous()
        hadamard_tensor = hadamard_tensor.view(-1,
                                                self.num_fields,
                                                self.embedding_dim,
                                                self.embedding_dim)
        V = torch.empty(hadamard_tensor.shape[0] * self.num_fields * self.interaction_units)
        V = (torch.masked_select(hadamard_tensor, self.upper_triange_mask))
        return V.contiguous().view(hadamard_tensor.shape[0], -1)


# Compressed interaction net layer
# This code is adapted from github repository:  https://github.com/xue-pai/FuxiCTR/blob/main/fuxictr/pytorch/layers/interaction.py
class CompressedInteractionNet(torch.nn.Module):
    def __init__(self, num_fields, embedding_dim, cin_layer_units, output_dim=1):
        super().__init__()
        self._num_fields = num_fields
        self._embedding_dim = embedding_dim
        self.cin_layer_units = cin_layer_units
        self.fc = torch.nn.Linear(sum(cin_layer_units), output_dim)
        self.cin_layer = torch.nn.ModuleDict()
        for i, unit in enumerate(self.cin_layer_units):
            in_channels = num_fields * self.cin_layer_units[i - 1] if i > 0 else num_fields ** 2
            out_channels = unit
            self.cin_layer["layer_" + str(i + 1)] = torch.nn.Conv1d(in_channels,
                                                                    out_channels,  # how many filters
                                                                    kernel_size=1) # kernel output shape

    def forward(self, inputs):
        inputs = inputs.reshape(-1, self._num_fields, self._embedding_dim)
        pooling_outputs = []
        X_0 = inputs
        batch_size = X_0.shape[0]
        embedding_dim = X_0.shape[-1]
        X_i = X_0
        for i in range(len(self.cin_layer_units)):
            hadamard_tensor = torch.einsum("bhd,bmd->bhmd", X_0, X_i)
            hadamard_tensor = hadamard_tensor.view( batch_size,-1, embedding_dim)
            X_i = self.cin_layer["layer_" + str(i + 1)](hadamard_tensor) \
                      .view(batch_size,-1, embedding_dim)
            pooling_outputs.append(X_i.sum(dim=-1))
        concate_vec = torch.cat(pooling_outputs, dim=-1)
        output = self.fc(concate_vec)
        return output

# Sequence Attention Layer
# This code is adapted from github repository:  https://github.com/RUCAIBox/RecBole/blob/master/recbole/model/layers.py
class DIEN_DIN_AttLayer(torch.nn.Module):
    def __init__(self, input_dim, att_hidden_size, att_activation, att_dropout, use_att_bn):
        super(DIEN_DIN_AttLayer, self).__init__()
        self.att_mlp_layers = MLPLayer(input_dim=input_dim*4, output_dim=1, hidden_units=att_hidden_size, hidden_activations=att_activation, dropout_rates=att_dropout, batch_norm=use_att_bn)

    def forward(self, query, keys, keys_length):
        batch_size, max_length, dim = keys.size()
        mask_mat = torch.arange(max_length).view(1, -1)
        query = query.unsqueeze(1).expand(batch_size, max_length, dim)
        input_tensor = torch.cat([query, keys, query-keys, query*keys], dim=-1)
        input_tensor = input_tensor.view(batch_size * max_length, -1) # [B*T 4*H]
        output = self.att_mlp_layers(input_tensor).view(batch_size, max_length) #[B T]
        mask = mask_mat.expand(output.shape[0], -1)
        mask = (mask >= keys_length.unsqueeze(1))
        mask_value = 0.0
        output = output.masked_fill(mask=mask, value=torch.tensor(mask_value))
        output = output.unsqueeze(1)
        output = output / (keys.shape[-1] ** 0.5)
        output = torch.nn.functional.softmax(output, dim=-1)
        output = torch.matmul(output, keys)
        return output

# Interest Evolving Layer
# This code is adapted from github repository:  https://github.com/RUCAIBox/RecBole/blob/master/recbole/model/sequential_recommender/dien.py
class InterestEvolvingLayer(torch.nn.Module):
    def __init__(
        self,
        embedding_size,
        gru_hidden_size,
        gru_num_layers,
        use_gru_bias,
        att_input_dim,
        att_hidden_units,
        att_activation,
        att_dropout,
        use_att_bn,
    ):
        super(InterestEvolvingLayer, self).__init__()
        self.attention_layer = DIEN_DIN_AttLayer(input_dim = att_input_dim,
                                                 att_hidden_size = att_hidden_units,
                                                 att_activation = att_activation,
                                                 att_dropout = att_dropout,
                                                 use_att_bn = use_att_bn)
        self.dynamic_rnn = torch.nn.GRU(
            input_size = embedding_size,
            hidden_size = gru_hidden_size,
            num_layers = gru_num_layers,
            bias = use_gru_bias,
            batch_first = True,
        )

    def forward(self, target_item, interest):
        packed_rnn_outputs,_= self.dynamic_rnn(interest)
        rnn_outputs,rnn_length = pad_packed_sequence(packed_rnn_outputs, batch_first=True)
        att_outputs = self.attention_layer(target_item, rnn_outputs, rnn_length)
        outputs = att_outputs.squeeze(1)
        return outputs

# Interest Extractor Network
# This code is adapted from github repository:  https://github.com/RUCAIBox/RecBole/blob/master/recbole/model/sequential_recommender/dien.py
class InterestExtractorNetwork(torch.nn.Module):
    def __init__(self,
                 embedding_size,
                 gru_hidden_size,
                 gru_num_layers,
                 use_gru_bias,
                 aux_input_dim,
                 aux_hidden_units,
                 aux_activation,
                 aux_dropout,
                 use_aux_bn):
        super(InterestExtractorNetwork, self).__init__()
        self.gru_layers = torch.nn.GRU(
            input_size = embedding_size,
            hidden_size = gru_hidden_size,
            num_layers = gru_num_layers,
            bias = use_gru_bias,
            batch_first = True,
        )
        self.auxiliary_net = MLPLayer(input_dim = aux_input_dim,
                                      output_dim = 1,
                                      hidden_units = aux_hidden_units,
                                      hidden_activations = aux_activation,
                                      final_activation = aux_activation,
                                      dropout_rates = aux_dropout,
                                      batch_norm = use_aux_bn)

    def forward(self, item_seq_pack, neg_item_seq_pack=None):
        packed_rnn_outputs,_=self.gru_layers(item_seq_pack)

        # padding all sequence
        rnn_outputs, _ = pad_packed_sequence(packed_rnn_outputs, batch_first=True)
        item_seq, _ = pad_packed_sequence(item_seq_pack, batch_first=True)
        neg_item_seq, _ = pad_packed_sequence(neg_item_seq_pack, batch_first=True)
        aux_loss = self.auxiliary_loss(rnn_outputs[:,:-1,:], item_seq[:,1:,:], neg_item_seq[:,1:,:])
        return packed_rnn_outputs, aux_loss

    def auxiliary_loss(self, h_states, click_seq, noclick_seq):
        click_input = torch.cat([h_states, click_seq], dim=-1)
        noclick_input = torch.cat([h_states, noclick_seq], dim=-1)

        click_prop = self.auxiliary_net(click_input.view(h_states.shape[0]*h_states.shape[1], -1)).view(-1, 1)
        click_target = torch.ones(click_prop.shape, device=click_input.device)

        noclick_prop = self.auxiliary_net(noclick_input.view(h_states.shape[0]*h_states.shape[1], -1)).view(-1, 1)
        noclick_target = torch.zeros(noclick_prop.shape, device=noclick_input.device)

        loss = F.binary_cross_entropy(
            torch.cat([click_prop, noclick_prop], dim=0),torch.cat([click_target, noclick_target],dim=0)
        )

        return loss


# Field-weighted factorization machine layer
# This code is adapted from github repository:  https://github.com/xue-pai/FuxiCTR/blob/main/fuxictr/pytorch/models/FwFM.py
class FwFMLayer(torch.nn.Module):
    def __init__(self,
                feature_count,
                embedding_dim):
        super().__init__()
        self.linear_weight_layer = torch.nn.Linear(feature_count * embedding_dim, 1, bias=False)
        self.inner_product_layer = InnerProductLayer(feature_count, embedding_dim)
        interact_dim = int(feature_count * (feature_count - 1) / 2)
        self.interaction_weight_layer = torch.nn.Linear(interact_dim, 1)

    def forward(self, inputs):
        linear_part = self.linear_weight_layer(inputs)
        inner_product_vec = self.inner_product_layer(inputs)
        poly2_part = self.interaction_weight_layer(inner_product_vec)
        return linear_part + poly2_part

# FeedForward
# This code is adapted from github repository:  https://github.com/RUCAIBox/RecBole/blob/master/recbole/model/layers.py
class FeedForward(torch.nn.Module):
    def __init__(self, hidden_size, inner_size, hidden_dropout_prob, hidden_act, layer_norm_eps):
        super(FeedForward, self).__init__()
        self.dense_1 = torch.nn.Linear(hidden_size, inner_size)
        self.intermediate_act_fn = self.get_hidden_act(hidden_act)

        self.dense_2 = torch.nn.Linear(inner_size, hidden_size)
        self.LayerNorm = torch.nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = torch.nn.Dropout(hidden_dropout_prob)

    def get_hidden_act(self, act):
        ACT2FN = {
            "gelu": self.gelu,
            "relu": torch.nn.functional.relu,
            "swish": self.swish,
            "tanh": torch.tanh,
            "sigmoid": torch.sigmoid,
        }
        return ACT2FN[act]

    def gelu(self, x):
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

    def swish(self, x):
        return x * torch.sigmoid(x)

    def forward(self, input_tensor):
        hidden_states = self.dense_1(input_tensor)
        hidden_states = self.intermediate_act_fn(hidden_states)

        hidden_states = self.dense_2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states

# TransformerLayer
# This code is adapted from github repository:  https://github.com/RUCAIBox/RecBole/blob/master/recbole/model/layers.py
class TransformerLayer(torch.nn.Module):

    def __init__(self, n_heads, hidden_size, intermediate_size, hidden_dropout_prob, hidden_act, layer_norm_eps):
        super(TransformerLayer, self).__init__()
        attention_head_size = int(hidden_size / n_heads)
        self.multi_head_attention = MultiHeadSelfAttention(
             input_dim=hidden_size, dim_per_head=attention_head_size, num_heads=n_heads, dropout_rate=hidden_dropout_prob,
            use_residual=True, layer_norm=True
        )
        self.feed_forward = FeedForward(hidden_size, intermediate_size, hidden_dropout_prob, hidden_act, layer_norm_eps)

    def forward(self, hidden_states):
        attention_output = self.multi_head_attention(hidden_states)
        feedforward_output = self.feed_forward(attention_output)
        return feedforward_output

#  TransformerEncoder
# This code is adapted from github repository:  https://github.com/RUCAIBox/RecBole/blob/master/recbole/model/layers.py
class TransformerEncoder(torch.nn.Module):
    def __init__(
        self,
        n_layers=2,
        n_heads=2,
        hidden_size=64,
        inner_size=256,
        hidden_dropout_prob=0.5,
        attn_dropout_prob=0.5,
        hidden_act='gelu',
        layer_norm_eps=1e-12
    ):

        super(TransformerEncoder, self).__init__()
        layer = TransformerLayer(n_heads, hidden_size, inner_size, hidden_dropout_prob, hidden_act, layer_norm_eps)
        self.layer = torch.nn.ModuleList([copy.deepcopy(layer) for _ in range(n_layers)])

    def forward(self, hidden_states):
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states)
        return hidden_states

