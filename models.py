import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, insize):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(insize, 2048)
        self.layer2 = nn.Linear(2048, 1024)
        self.layer3 = nn.Linear(1024, 512)
        self.layer4 = nn.Linear(512, 128)
        self.out = nn.Linear(128, 61)

        self.act_fn = nn.Sigmoid()
        # self.act_fn = nn.ReLU()
        self.soft = nn.Softmax()

    def forward(self, x):
        x = self.layer1(x)
        x = self.act_fn(x)

        x = self.layer2(x)
        x = self.act_fn(x)

        x = self.layer3(x)
        x = self.act_fn(x)

        x = self.layer4(x)
        x = self.act_fn(x)

        x = self.out(x)

        return x

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 64, 2, stride = 1)
        self.cbn1 = nn.BatchNorm2d(64)
        self.Mpool1 = nn.MaxPool2d(2, 2)
        self.cnn_layer = nn.Linear(6080, 4096)

        self.layer1 = nn.Linear(4096, 3064)
        self.layer2 = nn.Linear(3064, 2048)
        self.layer3 = nn.Linear(2048, 1024)
        self.layer4 = nn.Linear(1024, 512)

        self.out = nn.Linear(512, 61)

        self.act_fn1 = nn.ReLU()
        self.soft = nn.LogSoftmax(dim = 1)

        self.bn1 = nn.BatchNorm1d(num_features = 3064)
        self.bn2 = nn.BatchNorm1d(num_features = 2048)
        self.bn3 = nn.BatchNorm1d(num_features = 1024)
        self.bn4 = nn.BatchNorm1d(num_features = 512)

        self.dropout = nn.Dropout(0.2)

    def forward(self, x):

        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.act_fn1(x)
        x = self.cbn1(x)
        x = self.Mpool1(x)
        print(x.shape)
        x = x.view(-1, 6080)
        x = self.cnn_layer(x)


        x = self.layer1(x)
        x = self.act_fn1(x)
        x = self.bn1(x)
        x = self.dropout(x)

        x = self.layer2(x)
        x = self.act_fn1(x)
        x = self.bn2(x)
        x = self.dropout(x)

        x = self.layer3(x)
        x = self.act_fn1(x)
        x = self.bn3(x)
        x = self.dropout(x)

        x = self.layer4(x)
        x = self.act_fn1(x)
        x = self.bn4(x)
        x = self.dropout(x)

        x = self.out(x)
        x = self.soft(x)
        return x
