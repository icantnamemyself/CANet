import argparse
import os
import json
import pandas as pd

parser = argparse.ArgumentParser()
# dataset and dataloader args
parser.add_argument('--data_path', type=str, default='deginetica20.csv')
parser.add_argument('--save_path', type=str,
                    default='test')
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--max_len', type=int, default=20)
parser.add_argument('--train_batch_size', type=int, default=12)
parser.add_argument('--val_batch_size', type=int, default=12)
parser.add_argument('--test_batch_size', type=int, default=12)

# model args
parser.add_argument('--d_list', type=list, default=[64, 96, 128])
parser.add_argument('--emb_list', type=list, default=[64, 96, 128])
parser.add_argument('--layer_list', type=list, default=[2, 4, 6, 8])
parser.add_argument('--dropout', type=float, default=0.2)
parser.add_argument('--eval_per_steps', type=int, default=20000)
parser.add_argument('--enable_res_parameter', type=int, default=1)
parser.add_argument('--attn_heads', type=int, default=4)

# train args
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--lr_decay_rate', type=float, default=0.99)
parser.add_argument('--lr_decay_steps', type=int, default=20000)
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--num_epoch', type=int, default=60)
parser.add_argument('--metric_ks', type=list, default=[10, 20])
parser.add_argument('--best_metric', type=str, default='NDCG@10')

args = parser.parse_args()

DATA = pd.read_csv(args.data_path, header=None).values
num_item = DATA.max()
num_user = DATA.shape[0]
del DATA
args.num_item = int(num_item)
args.num_user = num_user
args.eval_per_steps = 4
args.lr_decay_steps = args.eval_per_steps // 2

if not os.path.exists(args.save_path):
    os.mkdir(args.save_path)

config_file = open(args.save_path + '/args.json', 'w')
tmp = args.__dict__
json.dump(tmp, config_file, indent=1)
print(args)
config_file.close()

