import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_normal_, uniform_, constant_
from .DynamicLayer import DynamicLinear, DynamicLayerNorm, gate_x, layer_constraint_x


class PositionalEmbedding(nn.Module):

    def __init__(self, max_len, d_model):
        super(PositionalEmbedding, self).__init__()

        # Compute the positional encodings once in log space.
        self.pe = nn.Embedding(max_len, d_model)

    def forward(self, x):
        batch_size = x.size(0)
        return self.pe.weight.unsqueeze(0).repeat(batch_size, 1, 1)


class Attention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """

    def forward(self, query, key, value, mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(query.size(-1))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn


class MultiHeadAttention(nn.Module):
    """
    Take in model size and number of heads.
    """

    def __init__(self, h, d_model, config, dropout=0.1, gate=False):
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0

        # We assume d_v always equals d_k
        self.max_d_model = d_model
        self.d_k = d_model // h
        self.h = h

        self.linear_layers = nn.ModuleList([DynamicLinear(d_model, d_model) for _ in range(3)])
        self.output_linear = DynamicLinear(d_model, d_model)
        self.attention = Attention()
        self.hidden_list = list_from_config(config, 'hidden')
        self.d_list = set([int(i / self.h) for i in self.hidden_list])
        self.hidden_tensor = torch.from_numpy(np.array(list(self.d_list))).float()
        self.gate = gate

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None, config=None, mode=None):
        batch_size = query.size(0)

        if mode == 'sample':
            if self.gate:
                d_k = config['hidden'] // self.h
                config_head = {'in': config['hidden'], 'out': config['hidden']}
            else:
                d_k = self.d_k
                config_head = {'in': self.max_d_model, 'out': self.max_d_model}
            # 1) Do all the linear projections in batch from d_model => h x d_k
            query, key, value = [l(x, config_head).view(batch_size, -1, self.h, d_k).transpose(1, 2)
                                 for l, x in zip(self.linear_layers, (query, key, value))]

            # 2) Apply attention on all the projected vectors in batch.
            x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)

            # 3) "Concat" using a view and apply a final linear.
            x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * d_k)

            return self.output_linear(x, config_head)
        else:
            config_head = {'in': self.max_d_model, 'out': self.max_d_model}
            query, key, value = [l(x, config_head).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                                 for l, x in zip(self.linear_layers, (query, key, value))]
            if self.gate:
                query, key, value = [gate_x(x, config, self.d_list) for x in [query, key, value]]
            x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)
            config_hidden = torch.matmul(config, self.hidden_tensor.to(x.device)).int().detach().cpu().numpy()
            x = self.attn_arrange(x, config_hidden)
            # x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
            x = self.output_linear(x, config_head)
            return x

    def attn_arrange(self, x, config: np.array):
        ret = torch.zeros(x.shape[0], x.shape[-2], self.h * self.d_k).to(x.device)
        x = x.transpose(1, 2).contiguous()
        for d in self.d_list:
            index = np.where(config == d)
            for h in range(self.h):
                ret[index, :, h * d:(h + 1) * d] += x[index, :, h, :d]
        return ret


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    """

    def __init__(self, size, enable_res_parameter, config, dropout=0.1, gate=False):
        super(SublayerConnection, self).__init__()
        self.norm = DynamicLayerNorm(size)
        self.max_d_model = size
        self.dropout = nn.Dropout(dropout)
        self.enable = enable_res_parameter
        if enable_res_parameter:
            self.a = nn.Parameter(torch.tensor(1e-8))
        self.hidden_list = list_from_config(config, 'hidden')
        self.hidden_tensor = torch.from_numpy(np.array(list(set(self.hidden_list)))).float()
        self.gate = gate

    def forward(self, x, sublayer, config, mode):
        "Apply residual connection to any sublayer with the same size."
        if self.gate:
            if mode == 'sample':
                config_norm = {'size': config['hidden']}
            else:
                config_norm = torch.matmul(config, self.hidden_tensor.to(x.device))
        else:
            config_norm = {'size': self.max_d_model}

        if not self.enable:
            return self.norm(x + self.dropout(sublayer(x, config, mode)), config_norm)
        else:
            return self.norm(x + self.dropout(self.a * sublayer(x, config, mode)), config_norm)


class PointWiseFeedForward(nn.Module):
    """
    FFN implement
    """

    def __init__(self, d_model, d_ffn, config, dropout=0.1, gate=False):
        super(PointWiseFeedForward, self).__init__()
        self.max_d_model = d_model
        self.max_d_ffn = d_ffn
        self.linear1 = DynamicLinear(d_model, d_ffn)
        self.linear2 = DynamicLinear(d_ffn, d_model)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.hidden_list = set(list_from_config(config, 'hidden'))
        self.gate = gate

    def forward(self, x, config, mode):
        if mode == 'sample':
            if self.gate:
                d_model = config['hidden']
                d_ffn = 4 * d_model
            else:
                d_model = self.max_d_model
                d_ffn = self.max_d_ffn
            config_1 = {'in': d_model, 'out': d_ffn}
            config_2 = {'in': d_ffn, 'out': d_model}
            return self.dropout(self.linear2(self.activation(self.linear1(x, config_1)), config_2))
        else:
            config_1 = {'in': self.max_d_model, 'out': self.max_d_ffn}
            config_2 = {'in': self.max_d_ffn, 'out': self.max_d_model}
            x = self.linear1(x, config_1)
            x = self.dropout(self.linear2(self.activation(x), config_2))
            if self.gate:
                x = gate_x(x, config, self.hidden_list)
            return x


class TransformerBlock(nn.Module):
    """
    TRM layer
    """

    def __init__(self, d_model, attn_heads, d_ffn, enable_res_parameter, config, dropout=0.1, gate=False):
        super(TransformerBlock, self).__init__()
        self.attn = MultiHeadAttention(attn_heads, d_model, config, dropout, gate)
        self.ffn = PointWiseFeedForward(d_model, d_ffn, config, dropout, gate)
        self.skipconnect1 = SublayerConnection(d_model, enable_res_parameter, config, dropout, gate)
        self.skipconnect2 = SublayerConnection(d_model, enable_res_parameter, config, dropout, gate)

    def forward(self, x, mask, config, mode):
        x = self.skipconnect1(x, lambda _x, _config, _mode: self.attn.forward(_x, _x, _x, mask=mask, config=_config,
                                                                              mode=_mode), config=config, mode=mode)
        x = self.skipconnect2(x, self.ffn, config, mode)
        return x


class DynamicBERT(nn.Module):
    """
    BERT model
    i.e., Embbedding + n * TRM + Output
    """

    def __init__(self, args):
        super(DynamicBERT, self).__init__()
        self.args = args
        self.num_item = args.num_item + 2
        self.d_model = args.d_list
        self.d_emb = args.emb_list
        self.layers = args.layer_list
        attn_heads = args.attn_heads
        dropout = args.dropout
        enable_res_parameter = args.enable_res_parameter
        self.device = args.device
        self.max_len = args.max_len
        self._form_config()

        # Embedding
        self.token = nn.Embedding(self.num_item, self.d_emb[-1])
        self.position = PositionalEmbedding(self.max_len, self.d_emb[-1])
        self.input_projection = DynamicLinear(self.d_emb[-1], self.d_model[-1])

        # TRMs
        self.TRMs = nn.ModuleList(
            [TransformerBlock(self.d_model[-1], attn_heads, 4 * self.d_model[-1], enable_res_parameter, self.config,
                              dropout,
                              gate=i > -1) for i in range(self.layers[-1])])

        # Output
        self.output = DynamicLinear(self.d_model[-1], self.num_item - 1)
        self.attention_mask = torch.tril(torch.ones((self.max_len, self.max_len), dtype=torch.bool)).to(self.device)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            stdv = np.sqrt(1. / self.num_item)
            uniform_(module.weight.data, -stdv, stdv)
        elif isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0.1)

    def _form_config(self):
        self.config = {}
        index = 0
        self.layer_list = []
        self.emb_list = []
        self.hidden_list = []
        for hidden in self.d_model:
            for l in self.layers:
                for emb in self.d_emb:
                    self.config[index] = {'layer': l, 'emb': emb, 'hidden': hidden}
                    index += 1
                    self.layer_list.append(l)
                    self.emb_list.append(emb)
                    self.hidden_list.append(hidden)
        self.layer_tensor = torch.from_numpy(np.array(self.layers)).float().to(self.device)
        self.emb_tensor = torch.from_numpy(np.array(self.emb_list)).float().to(self.device)
        self.hidden_tensor = torch.from_numpy(np.array(self.hidden_list)).float().to(self.device)
        print(self.config)

    def get_config_num(self):
        return len(self.config), self.d_model, self.layers

    def forward(self, x, config_choice):
        self.attention_mask = torch.tril(torch.ones((self.max_len, self.max_len), dtype=torch.bool)).to(x.device)
        self.layer_tensor = self.layer_tensor.to(x.device)
        self.emb_tensor = self.emb_tensor.to(x.device)
        self.hidden_tensor = self.hidden_tensor.to(x.device)
        mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)
        mask *= self.attention_mask
        try:
            mode = 'dynamic' if isinstance(config_choice[0], torch.Tensor) else 'sample'
        except:
            mode = 'sample'

        if mode == 'dynamic':
            config = config_choice
            embs_set = self.d_emb
            config_emb = torch.zeros(x.shape[0], len(embs_set)).to(x.device)
            for index, emb in enumerate(self.emb_list):
                config_emb[:, embs_set.index(emb)] += config[:, index]
            hidden_set = self.d_model
            config_hidden = torch.zeros(x.shape[0], len(hidden_set)).to(x.device)
            for index, hidden in enumerate(self.hidden_list):
                config_hidden[:, hidden_set.index(hidden)] += config[:, index]
            layer_set = self.layers
            config_layer = torch.zeros(x.shape[0], len(layer_set)).to(x.device)
            for index, layer in enumerate(self.layer_list):
                config_layer[:, layer_set.index(layer)] += config[:, index]
        else:
            config = self.config[config_choice]

        x = self.token(x) + self.position(x)  # B * L --> B * L * D

        if mode == 'sample':
            x = x[:, :, :config['emb']]
            config_input = {'in': config['emb'], 'out': config['hidden']}
            x = self.input_projection(x, config_input)
            for index, TRM in enumerate(self.TRMs):
                if index == config['layer']:
                    break
                x = TRM(x, mask, config, mode)

            config_linear = {'in': config['hidden'], 'out': self.num_item - 1}
            return self.output(x, config_linear)  # B * L * D --> B * L * N

        else:
            x = gate_x(x, config_emb, embs_set)
            config_input = {'in': self.d_emb[-1], 'out': self.d_model[-1]}
            x = self.input_projection(x, config_input)
            x = gate_x(x, config_hidden, hidden_set)
            for index, TRM in enumerate(self.TRMs):
                if index < self.layers[0]:
                    x = TRM(x, mask, config_hidden, mode)
                else:
                    x_new = TRM(x, mask, config_hidden, mode)
                    x = layer_constraint_x(x_new, x, index, self.layers, config_layer)

            config_linear = {'in': self.d_model[-1], 'out': self.num_item - 1}
            return self.output(x, config_linear)  # B * L * D --> B * L * N


def list_from_config(config: dict, element):
    ret_list = []
    for k, v in config.items():
        ret_list.append(v[element])
    return ret_list
