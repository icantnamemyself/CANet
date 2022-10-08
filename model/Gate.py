import torch
import torch.nn as nn
import numpy as np
from torch.nn.init import xavier_normal_, uniform_, constant_
from .BERT import PositionalEmbedding, TransformerBlock


class Gate(nn.Module):
    def __init__(self, args, num_class_emb=4, num_class_depth=4):
        super(Gate, self).__init__()
        self.num_item = args.num_item + 2
        d_model = 32
        attn_heads = args.attn_heads
        d_ffn = 128
        layers = 1
        dropout = args.dropout
        enable_res_parameter = args.enable_res_parameter
        self.device = args.device
        self.max_len = args.max_len
        self.attention_mask = torch.tril(torch.ones((self.max_len, self.max_len), dtype=torch.bool)).to(self.device)

        # Embedding
        self.token = nn.Embedding(self.num_item, d_model)
        self.position = PositionalEmbedding(self.max_len, d_model)

        # TRMs
        self.TRMs = nn.ModuleList(
            [TransformerBlock(d_model, attn_heads, d_ffn, enable_res_parameter, dropout) for i in range(layers)])

        # Output
        self.output_emb = nn.Linear(d_model, num_class_emb)
        self.output_depth = nn.Linear(d_model, num_class_depth)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            stdv = np.sqrt(1. / self.num_item)
            uniform_(module.weight.data, -stdv, stdv)
        elif isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0.1)

    def forward(self, x):

        mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)

        x = self.token(x) + self.position(x)

        for index, TRM in enumerate(self.TRMs):
            x = TRM(x, mask)

        return self.output_emb(x[:, -1, :]), self.output_depth(x[:, -1, :])
