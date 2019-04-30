import argparse

import numpy as np
import torch
import torch.optim as optim
import torch.utils.data as data
from torch.nn import TripletMarginLoss
from tqdm import tqdm

from dataloader import Dataset, collect_fn, negative_batch_generator
from model import UAEModel
from utils import load_model

torch.backends.cudnn.benchmark = True


def main():
    train_dataset = Dataset()
    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=collect_fn,
                                   shuffle=True, num_workers=args.workers, pin_memory=True)
    neg_gen = negative_batch_generator(args.max_len, args.batch_size, args.neg_size)
    train(train_loader, neg_gen)


def train(train_loader, neg_gen):
    model = UAEModel(device).to(device)
    if args.warm_up:
        model = load_model(args.max_len, device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_function = TripletMarginLoss()
    min_loss = np.inf
    for epoch in range(args.epoch):
        print('%d / %d Epoch' % (epoch, args.epoch))
        epoch_loss = train_epoch(train_loader, neg_gen, model, optimizer, loss_function)
        print(epoch_loss)
        if epoch_loss < min_loss:
            min_loss = epoch_loss
            torch.save(model.state_dict(), 'model.bin')
    return model


def train_epoch(train_loader, neg_gen, model, optimizer, loss_function):
    total_loss = 0
    model.train()
    model.mode = 'train'
    for i, batch in enumerate(tqdm(train_loader)):
        optimizer.zero_grad()
        tokens = batch.tokens.to(device)
        sentence_embs = batch.embedings.to(device)
        neg_bags = next(neg_gen).to(device)
        if tokens.size(0) < neg_bags.size(0):
            neg_bags = neg_bags[:tokens.size(0)]
        pos, neg, anchor = model(tokens, sentence_embs, neg_bags)
        loss = loss_function(anchor, pos, neg) / args.batch_size
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    return total_loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Unsupervised Aspect Extraction')
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--epoch', default=1000, type=int)
    parser.add_argument('--workers', default=2, type=int)
    parser.add_argument('--max-len', default=64, type=int)
    parser.add_argument('--neg-size', default=20, type=int)
    parser.add_argument('--cuda', default=True, type=bool)
    parser.add_argument('--warm-up', default=False, type=bool)
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--weight_decay', default=0, type=float)

    args = parser.parse_args()
    if args.cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    main()
