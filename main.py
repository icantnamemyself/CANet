from dataset import TrainDataset, EvalDataset
from CANet import CANet
from process import Trainer
from args import args
import torch.utils.data as Data


def main():
    train_dataset = TrainDataset(args.max_len, args.data_path, device=args.device)
    train_loader = Data.DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)

    val_dataset = EvalDataset(args.max_len, mode='val', path=args.data_path, device=args.device, )
    val_loader = Data.DataLoader(val_dataset, batch_size=args.val_batch_size)

    test_dataset = EvalDataset(args.max_len, mode='test', path=args.data_path, device=args.device)
    test_loader = Data.DataLoader(test_dataset, batch_size=args.test_batch_size)
    print('dataset initial ends')

    model = CANet(args)
    print('model initial ends')

    trainer = Trainer(args, model, train_loader, val_loader, test_loader)
    print('train process ready')

    trainer.train()


if __name__ == '__main__':
    main()
