import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class DynamicLinear(nn.Module):
    def __init__(self, max_input, max_output):
        super(DynamicLinear, self).__init__()
        self.layer = nn.Linear(max_input, max_output)
        self.max_input = max_input
        self.max_output = max_output

    def forward(self, x, config):
        input_dim = config['in']
        output_dim = config['out']
        if input_dim == self.max_input and output_dim == self.max_output:
            return F.linear(x, self.layer.weight, bias=self.layer.bias)
        return F.linear(x, self.layer.weight[:output_dim, :input_dim], bias=self.layer.bias[:output_dim])


class DynamicLayerNorm(nn.Module):
    def __init__(self, max_size, eps=1e-5):
        super(DynamicLayerNorm, self).__init__()
        self.max_size = (max_size,)
        self.weight = Parameter(torch.ones(self.max_size, ))
        self.bias = Parameter(torch.zeros(self.max_size, ))
        self.eps = eps

    def forward(self, x, config):
        mode = 'sample' if isinstance(config, dict) else 'dynamic'
        if mode == 'sample':
            size = config['size']
            if size == self.max_size[0]:
                return F.layer_norm(x, (size,), self.weight, self.bias, self.eps)
            return F.layer_norm(x, (size,), self.weight[:size], self.bias[:size], self.eps)
        else:
            ret = torch.zeros_like(x).to(x.device)
            for index, _config in enumerate(config):
                size = _config.int()
                if size == self.max_size[0]:
                    ret[index] = F.layer_norm(x[index], (size,), self.weight, self.bias, self.eps)
                else:
                    ret[index, :, :size] = F.layer_norm(x[index, :, :size], (size,), self.weight[:size],
                                                        self.bias[:size], self.eps)
            return ret


def gate_x(x, choices, embs):
    # mask x for batch operation in gate training
    ret = torch.zeros_like(x).to(x.device)
    if len(x.shape) == 3:
        for index, emb in enumerate(embs):
            ret[:, :, :emb] += x[:, :, :emb] * (choices[:, index, None, None].expand_as(x[:, :, :emb]))
    else:
        for index, emb in enumerate(embs):
            ret[:, :, :, :emb] += x[:, :, :, :emb] * (
                choices[:, index, None, None, None].expand_as(x[:, :, :, :emb]))
    return ret


def layer_constraint_x(x, x_old, layer_index, depths, choices):
    ret = torch.zeros_like(x).to(x.device)
    for index, depth in enumerate(depths):
        if depth > layer_index:
            ret += x * choices[:, index, None, None].expand_as(x)
        else:
            ret += x_old * choices[:, index, None, None].expand_as(x)
    return ret


def gumbel_softmax(prob, tau=1, dim=-1):
    logits = torch.log(torch.softmax(prob, dim=-1))
    gumbels = -torch.empty_like(logits,
                                memory_format=torch.legacy_contiguous_format).exponential_().log()  # ~Gumbel(0,1)
    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
    y_soft = gumbels.softmax(dim)
    with torch.no_grad():
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
    ret = y_hard - y_soft.detach() + y_soft
    return ret, index
