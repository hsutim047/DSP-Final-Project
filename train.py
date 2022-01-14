import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from dataset import TIMITDataset
from models import MLP
import argparse
from methods import *
from utils import *


def operate(X, args):
    if args.method == 'concatenate':
        return concatenate(X, args.window_size)
    elif args.method == 'average':
        return average(X, args.window_size)
    elif args.method == 'left_concatenate':
        return left_concatenate(X, args.window_size)
    elif args.method == 'right_concatenate':
        return right_concatenate(X, args.window_size)
    elif args.method == 'weight_average':
        return weight_average(X, args.window_size)
    else:
        return X


def get_input_size(method, ws):
    if method == 'concatenate':
        insize = (2 * ws + 1) * 39
    elif method == 'average':
        insize = 39
    elif method == 'left_concatenate' or sys.argv[2] == 'right_concatenate':
        insize = (ws + 1) * 39
    elif method == 'weight_average':
        insize = 39
    else:
        insize = 39
    return insize


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--window_size', type = int, default = 5)
    parser.add_argument('--data_prefix', type = str, default = './_preprocess_data/')
    parser.add_argument('--method', type = str, default = 'concatenate')
    parser.add_argument('--batch_size', type = int, default = 64)
    parser.add_argument('--seed', type = int, default = 0)
    parser.add_argument('--learning_rate', type = float, default = 1e-4)
    parser.add_argument('--epochs', type = int, default = 40)
    return parser


def train(model, train_loader, device, criterion, optimizer, train_acc_loss):
    [train_loss, train_acc] = train_acc_loss

    model.train()
    for _, data in enumerate(train_loader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        batch_loss = criterion(outputs, labels)
        _, prediction = torch.max(outputs, 1)
        batch_loss.backward()
        optimizer.step()

        train_acc += (prediction.cpu() == labels.cpu()).sum().item()
        train_loss += batch_loss.item()

    return [train_acc, train_loss]


def test(model, test_loader, device, criterion, optimizer, test_acc_loss):
    [test_acc, test_loss] = test_acc_loss

    model.eval()
    with torch.no_grad():
        for _, data in enumerate(test_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            batch_loss = criterion(outputs, labels)
            _, prediction = torch.max(outputs, 1)

            test_acc += (prediction.cpu() == labels.cpu()).sum().item() # get the index of the class with the highest probability
            test_loss += batch_loss.item()

    return [test_acc, test_loss]


def main():

    """Reference
    https://colab.research.google.com/github/ga642381/ML2021-Spring/blob/main/HW02/HW02-1.ipynb
    """

    parser = get_parser()
    args = parser.parse_args()
    fix_seed(args.seed)

    data_prefix = args.data_prefix
    print('Loading data')
    train_x, train_y = operate(np.load(data_prefix + 'train.npy'), args), np.load(data_prefix + 'train_label.npy')
    test_x, test_y = operate(np.load(data_prefix + 'test.npy'), args), np.load(data_prefix + 'test_label.npy')

    print('Size of training data: {}'.format(train_x.shape, train_y.shape))
    print('Size of testing data: {}'.format(test_x.shape, test_y.shape))

    train_set = TIMITDataset(train_x, train_y)
    test_set = TIMITDataset(test_x, test_y)
    train_loader = DataLoader(train_set, batch_size = args.batch_size, shuffle = True)
    test_loader = DataLoader(test_set, batch_size = args.batch_size, shuffle = False)

    device = get_device()

    input_size = get_input_size(args.method, args.window_size)

    model = MLP(input_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = args.learning_rate)

    print('Start training')

    best_acc = 0.0
    for epoch in range(args.epochs):
        [train_acc, train_loss, test_acc, test_loss] = [0.0] * 4

        [train_acc, train_loss] = train(model, train_loader, device, criterion, optimizer, [train_acc, train_loss])

        if len(test_set) > 0:
            [test_acc, test_loss] = test(model, test_loader, device, criterion, optimizer, [test_acc, test_loss])

            best_acc = max(best_acc, test_acc / len(test_set))

            print('[{:03d}/{:03d}] Train Acc: {:3.6f} Loss: {:3.6f} | Val Acc: {:3.6f} loss: {:3.6f} | Best Accuracy: {:3.6f}'.format(
                epoch + 1,
                args.epochs,
                train_acc / len(train_set),
                train_loss / len(train_loader),
                test_acc / len(test_set),
                test_loss / len(test_loader),
                best_acc
            ))


if __name__ == '__main__':
    main()
