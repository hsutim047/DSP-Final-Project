import gc
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from dataset import TIMITDataset
from models import Classifier
from methods import *
from utils import *

# (time CUDA_VISIBLE_DEVICES=0 python3 train.py 2 right_concatenate) &> logs/right_concatenate_2.log

def operate(X, ws = 1):
    if sys.argv[2] == 'concatenate':
        return concatenate(X, ws)
    elif sys.argv[2] == 'average':
        return average(X, ws)
    elif sys.argv[2] == 'left_concatenate':
        return left_concatenate(X, ws)
    elif sys.argv[2] == 'right_concatenate':
        return right_concatenate(X, ws)
    elif sys.argv[2] == 'weight_average':
        return weight_average(X, ws)
    else:
        return X

print('Loading data ...')

ws = int(sys.argv[1])

data_root='/data1/ytsanhsuu/preprocess_data/'
train_x, train_y = operate(np.load(data_root + 'train.npy'), ws), np.load(data_root + 'train_label.npy')
test_x, test_y = operate(np.load(data_root + 'test.npy'), ws), np.load(data_root + 'test_label.npy')

print('Size of training data: {}'.format(train_x.shape, train_y.shape))
print('Size of testing data: {}'.format(test_x.shape, test_y.shape))

BATCH_SIZE = 64

train_set = TIMITDataset(train_x, train_y)
val_set = TIMITDataset(test_x, test_y)
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True) #only shuffle the training data
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

# fix random seed for reproducibility
same_seeds(0)

# get device
device = get_device()
print(f'DEVICE: {device}')

# training parameters
num_epoch = 40               # number of training epoch
learning_rate = 0.0001       # learning rate

# the path where checkpoint saved
model_path = './model.ckpt'

if sys.argv[2] == 'concatenate':
    insize = (2 * ws + 1) * 39
elif sys.argv[2] == 'average':
    insize = 39
elif sys.argv[2] == 'left_concatenate' or sys.argv[2] == 'right_concatenate':
    insize = (ws + 1) * 39
elif sys.argv[2] == 'weight_average':
    insize = 39
else:
    insize = 39

# create model, define a loss function, and optimizer
model = Classifier(insize).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# start training

best_acc = 0.0
for epoch in range(num_epoch):
    train_acc = 0.0
    train_loss = 0.0
    val_acc = 0.0
    val_loss = 0.0

    # training
    model.train() # set the model to training mode
    for i, data in enumerate(train_loader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        batch_loss = criterion(outputs, labels)
        _, train_pred = torch.max(outputs, 1) # get the index of the class with the highest probability
        batch_loss.backward()
        optimizer.step()

        train_acc += (train_pred.cpu() == labels.cpu()).sum().item()
        train_loss += batch_loss.item()

    # validation
    if len(val_set) > 0:
        model.eval() # set the model to evaluation mode
        with torch.no_grad():
            for i, data in enumerate(val_loader):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                batch_loss = criterion(outputs, labels)
                _, val_pred = torch.max(outputs, 1)

                val_acc += (val_pred.cpu() == labels.cpu()).sum().item() # get the index of the class with the highest probability
                val_loss += batch_loss.item()

            print('[{:03d}/{:03d}] Train Acc: {:3.6f} Loss: {:3.6f} | Val Acc: {:3.6f} loss: {:3.6f}'.format(
                epoch + 1, num_epoch, train_acc/len(train_set), train_loss/len(train_loader), val_acc/len(val_set), val_loss/len(val_loader)
            ))

            # if the model improves, save a checkpoint at this epoch
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), model_path)
                print('saving model with acc {:.3f}'.format(best_acc/len(val_set)))
    else:
        print('[{:03d}/{:03d}] Train Acc: {:3.6f} Loss: {:3.6f}'.format(
            epoch + 1, num_epoch, train_acc/len(train_set), train_loss/len(train_loader)
        ))

# if not validating, save the last epoch
if len(val_set) == 0:
    torch.save(model.state_dict(), model_path)
    print('saving model at last epoch')
