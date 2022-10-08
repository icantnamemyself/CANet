import time
import torch
import torch.nn as nn
from tqdm import tqdm
from loss import CE
from torch.optim.lr_scheduler import LambdaLR


class Trainer():
    def __init__(self, args, model, train_loader, val_loader, test_loader):
        self.args = args
        self.device = args.device
        print(self.device)
        self.model = model.to(torch.device(self.device))

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.lr_decay = args.lr_decay_rate
        self.lr_decay_steps = args.lr_decay_steps

        self.cr = CE(self.model)
        self.test_loss = nn.CrossEntropyLoss(ignore_index=0)

        self.num_epoch = args.num_epoch
        self.epoch = 0
        self.metric_ks = args.metric_ks
        self.eval_per_steps = args.eval_per_steps
        self.save_path = args.save_path
        if self.num_epoch:
            self.result_file = open(self.save_path + '/result.txt', 'w')
            self.result_file.close()

        self.step = 0
        self.metric = args.best_metric
        self.best_metric = -1e9

        self.labels = torch.zeros(512, self.args.num_item + 1).to(self.device)

    def train(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        self.scheduler = LambdaLR(self.optimizer, lr_lambda=lambda step: self.lr_decay ** step, verbose=True)
        for epoch in range(self.num_epoch):
            loss_epoch, time_cost= self._train_one_epoch()
            self.result_file = open(self.save_path + '/result.txt', 'a+')
            print('Model train epoch:{0},loss:{1},training_time:{2}'.format(epoch + 1, loss_epoch,
                                                                                             time_cost))
            print('Model train epoch:{0},loss:{1},training_time:{2}'.format(epoch + 1, loss_epoch,
                                                                                             time_cost),
                  file=self.result_file)
            self.result_file.close()

    def _train_one_epoch(self):
        self.epoch += 1
        optim = self.optimizer
        t0 = time.perf_counter()
        self.model.train()
        tqdm_dataloader = tqdm(self.train_loader)

        loss_sum = 0
        for idx, batch in enumerate(tqdm_dataloader):
            batch = [x.to(self.device) for x in batch]

            optim.zero_grad()
            loss = self.cr.compute(batch)
            loss_sum += loss.cpu().item()
            loss.backward()
            optim.step()

            self.step += 1
            if self.step % self.lr_decay_steps == 0:
                self.scheduler.step()
                self.cr.step()
                metric = {}
                for mode in ['test']:
                    depth, metric[mode] = self.eval_model(mode)
                print(metric, depth)
                self.result_file = open(self.save_path + '/result.txt', 'a+')
                print('step{0}'.format(self.step), file=self.result_file)
                print(metric, file=self.result_file)
                print(depth, file=self.result_file)
                self.result_file.close()
                if metric['test'][self.metric] > self.best_metric:
                    torch.save(self.model.state_dict(), self.save_path + '/model.pkl')
                    self.result_file = open(self.save_path + '/result.txt', 'a+')
                    print('saving model of step{0}'.format(self.step), file=self.result_file)
                    self.result_file.close()
                    self.best_metric = metric['test'][self.metric]
                self.model.train()

        return loss_sum / idx, time.perf_counter() - t0

    def eval_model(self, mode):
        self.model.eval()
        tqdm_data_loader = tqdm(self.val_loader) if mode == 'val' else tqdm(self.test_loader)
        metrics = {}
        depth_dict = {i: 0 for i in range(self.model.config_num)}

        with torch.no_grad():
            for idx, batch in enumerate(tqdm_data_loader):
                batch = [x.to(self.device) for x in batch]
                metrics_batch = self.compute_metrics(batch)
                if len(metrics_batch) == 2:
                    metrics_batch, index = metrics_batch
                    for i in index:
                        depth_dict[i.item()] += 1

                for k, v in metrics_batch.items():
                    if not metrics.__contains__(k):
                        metrics[k] = v
                    else:
                        metrics[k] += v

        for k, v in metrics.items():
            metrics[k] = v / (idx + 1)
        return depth_dict, metrics

    def compute_metrics(self, batch):
        seqs, answers = batch
        ret = self.model(seqs, 'test')

        scores, choices = ret
        index = choices.cpu().int()
        scores = scores[:, -1, :]
        test_loss = self.test_loss(scores.view(-1, scores.shape[-1]), answers.view(-1))
        row = []
        col = []
        seqs = seqs.tolist()
        answers = answers.tolist()
        for i in range(len(answers)):
            seq = list(set(seqs[i] + answers[i]))
            seq.remove(answers[i][0])
            if self.args.num_item + 1 in seq:
                seq.remove(self.args.num_item + 1)
            row += [i] * len(seq)
            col += seq
            self.labels[i][answers[i]] = 1
        scores[row, col] = -1e9
        metrics = recalls_and_ndcgs_for_ks(scores, self.labels[:len(seqs)], self.metric_ks)
        metrics['test_loss'] = test_loss.item()
        self.labels[self.labels == 1] = 0
        return metrics, index


def recalls_and_ndcgs_for_ks(scores, labels, ks):
    metrics = {}

    answer_count = labels.sum(1)

    labels_float = labels.float()
    rank = (-scores).argsort(dim=1)
    cut = rank
    for k in sorted(ks, reverse=True):
        cut = cut[:, :k]
        hits = labels_float.gather(1, cut)
        metrics['Recall@%d' % k] = \
            (hits.sum(1) / torch.min(torch.Tensor([k]).to(labels.device),
                                     labels.sum(1).float())).mean().cpu().item()

        position = torch.arange(2, 2 + k)
        weights = 1 / torch.log2(position.float())
        dcg = (hits * weights.to(hits.device)).sum(1)
        idcg = torch.Tensor([weights[:min(int(n), k)].sum() for n in answer_count]).to(dcg.device)
        ndcg = (dcg / idcg).mean()
        metrics['NDCG@%d' % k] = ndcg.cpu().item()

    return metrics
