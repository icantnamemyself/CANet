import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


def gen_softlabel(num, multipier=1):
    weight_list = [2 ** i / 2 ** num for i in range(num)]
    weight = np.array(weight_list) + 1e-8
    label_1 = torch.softmax(torch.tensor(np.log(weight) * multipier), dim=-1)
    weight_list.reverse()
    weight = np.array(weight_list) + 1e-8
    label_0 = torch.softmax(torch.tensor(np.log(weight) * multipier), dim=-1)
    return label_0, label_1


class CE:
    def __init__(self, model):
        self.model = model
        self.ce_d = nn.CrossEntropyLoss(reduction='none', ignore_index=0)
        self.ce_s = nn.CrossEntropyLoss(ignore_index=0)
        self.ce_kd = nn.CrossEntropyLoss() if int(torch.__version__.split('.')[1]) >= 10 else CrossEntropy()
        self.mse = nn.MSELoss()
        self.hit_sum = 0

        self.path_num = self.model.config_num
        self.d_emb = self.model.backbone.d_emb
        self.d_model = self.model.backbone.d_model
        self.layers = self.model.backbone.layers
        self.emb_list = self.model.backbone.emb_list
        self.hidden_list = self.model.backbone.hidden_list
        self.layer_list = self.model.backbone.layer_list

        # soft labels
        label_0, label_1 = gen_softlabel(len(self.d_emb), multipier=2)
        self.emb_label = torch.stack([label_0, label_1], dim=0).float()

        label_0, label_1 = gen_softlabel(len(self.d_model), multipier=2)
        self.hidden_label = torch.stack([label_0, label_1], dim=0).float()

        label_0, label_1 = gen_softlabel(len(self.layers), multipier=1)
        self.layer_label = torch.stack([label_0, label_1], dim=0).float()

        self.uniform_label = torch.ones(self.path_num) / self.path_num

        self.auxiliary_weight_uniform = 0.01
        self.auxiliary_weight_guide = 0.01

    def compute(self, batch):
        seqs, labels = batch

        # one-step prediction
        with torch.no_grad():
            output_smallest = self.model(seqs, mode='smallest')
            score = output_smallest[:, -1, :]
            answers = labels[:, -1].tolist()
            labels_tmp = torch.zeros_like(score).to(score.device)
            row = []
            col = []
            seqs_list = seqs.tolist()
            for i in range(len(answers)):
                seq = list(set(seqs_list[i]))
                row += [i] * len(seq)
                col += seq
                labels_tmp[i][answers[i]] = 1
            score[row, col] = -1e9
            labels_float = labels_tmp.float()
            rank = (-score).argsort(dim=1)
            cut = rank
            cut = cut[:, :10] # k, hyper-paramter
            hits = labels_float.gather(1, cut).sum(1)
            self.hit_sum += torch.sum(hits)

            # soft labels generation
            emb_label = torch.zeros(hits.shape[0], len(self.d_emb))
            emb_label[hits > 0] = self.emb_label[0]
            emb_label[hits == 0] = self.emb_label[1]
            hidden_label = torch.zeros(hits.shape[0], len(self.d_model))
            hidden_label[hits > 0] = self.hidden_label[0]
            hidden_label[hits == 0] = self.hidden_label[1]
            layer_label = torch.zeros(hits.shape[0], len(self.layers))
            layer_label[hits > 0] = self.layer_label[0]
            layer_label[hits == 0] = self.layer_label[1]

        # L_SR
        outputs, prob = self.model(seqs)
        outputs = outputs.view(-1, outputs.shape[-1])
        labels = labels.view(-1)
        loss = self.ce_s(outputs, labels)

        # p(emb|u) etc.
        emb_prob, hidden_prob, layer_prob = self.gen_decoupled_probs_for_input(prob)
        # L_Guide
        guide_loss = self.ce_kd(emb_prob, emb_label.to(prob.device)) + self.ce_kd(hidden_prob, hidden_label.to(
            prob.device)) + self.ce_kd(layer_prob, layer_label.to(prob.device))
        loss += self.auxiliary_weight_guide * guide_loss
        # L_Uni
        uniform_loss = self.ce_kd(prob, self.uniform_label.repeat(prob.shape[0], 1).to(prob.device))
        loss += self.auxiliary_weight_uniform * uniform_loss

        return loss

    def gen_decoupled_probs_for_input(self, probs):
        emb_prob = torch.zeros(probs.shape[0], len(self.d_emb)).to(probs.device)
        hidden_prob = torch.zeros(probs.shape[0], len(self.d_model)).to(probs.device)
        layer_prob = torch.zeros(probs.shape[0], len(self.layers)).to(probs.device)
        for i in range(probs.shape[1]):
            prob_path = probs[:, i]
            emb_prob[:, self.d_emb.index(self.emb_list[i])] = emb_prob[:,
                                                              self.d_emb.index(self.emb_list[i])] + prob_path
            hidden_prob[:, self.d_model.index(self.hidden_list[i])] = hidden_prob[:, self.d_model.index(
                self.hidden_list[i])] + prob_path
            layer_prob[:, self.layers.index(self.layer_list[i])] = layer_prob[:,
                                                                   self.layers.index(self.layer_list[i])] + prob_path
        return emb_prob, hidden_prob, layer_prob

    def step(self):
        result_file = open(self.model.backbone.args.save_path + '/result.txt', 'a')
        print(self.hit_sum)
        print(self.hit_sum, file=result_file)
        result_file.close()
        self.hit_sum = 0


class CrossEntropy(nn.Module):
    def __init__(self):
        super(CrossEntropy, self).__init__()

    def forward(self, input, target):
        input = F.log_softmax(input, dim=-1)
        loss = (- input * target).sum() / input.shape[0]
        return loss
