import torch
import torch.nn as nn
from model.DynamicBert import DynamicBERT
from model.Gate import Gate
from model.DynamicLayer import gumbel_softmax


class CANet(nn.Module):
    def __init__(self, args):
        super(CANet, self).__init__()
        self.backbone = DynamicBERT(args)
        self.num_item = args.num_item + 1
        self.max_len = args.max_len
        self.device = args.device
        self.config_num, self.emb_num, self.depth_num = self.backbone.get_config_num()
        self.router = Gate(args, self.config_num, 1)

    def forward(self, x, mode='train'):

        if mode == 'train':
            config_prob, _ = self.router(x)
            config_choices, index = gumbel_softmax(config_prob)
            pred = self.backbone(x, config_choices)
            return pred, config_prob

        elif mode == 'test':
            config_prob, _ = self.router(x)
            config_choices, index = gumbel_softmax(config_prob)
            pred = self.backbone(x, config_choices)
            return pred, index

        elif mode == 'smallest':
            config_choices = 0
            pred = self.backbone(x, config_choices)
            return pred
