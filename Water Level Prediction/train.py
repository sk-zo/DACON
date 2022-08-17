import sys
import pickle
import argparse
import torch
from torch.utils.data import DataLoader
from model import *
from Trainer import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', default="s", help="('SimpleRegressor':s)")
    parser.add_argument('--d_model', default=10, type=int, help="input feature dimension")
    parser.add_argument('--num_label', default=4, type=int, help="lbael feature length")
    parser.add_argument('--seq_len', default=720, type=int, help="data sequence length")
    parser.add_argument('--batch_size', default=64, type=int, help="batch size")
    parser.add_argument('--max_epoch', default=20, type=int, help="num of epochs of training")
    parser.add_argument('--device', default=torch.device('cuda'), help="torch.device object")
    parser.add_argument('--max_grad_norm', default=1, type=float, help="max gradient norm")
    parser.add_argument('--lr', default=5e-4, type=float, help="learning rate")
    parser.add_argument('--data_path', default='dataset', help='data path')
    parser.add_argument('--model_save_path', default='trained_model/base', help='model save path')

    args = parser.parse_args()

    if args.model == "s":
        with open(args.data_path, 'rb') as f:
            dataset = pickle.load(f)
            dataloader = DataLoader(dataset, args.batch_size)
            model = SimpleRegressor(input_len=dataset.data.size(1).to(args.device)
            trainer = Trainer(args, model, dataloader)

    trainer.fit(eval=False)
    trainer.save_model()
