import time
import torch
import torch.nn as nn
import torch.nn.functional as F

# modes and num of CNN layers
# 1 --> 3
# 2 --> 5
# 3 --> 7

class ConvNet(nn.Module):
    def __init__(self, mode):
        super(ConvNet, self).__init__() 
        self.mode =  mode

        # define model architecture 
        # self.cnn_layers = [
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32 * 2, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32 * 2, out_channels=64 * 2, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64 * 2, out_channels=64 * 2, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64 * 2, out_channels=128 * 2, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels=128 * 2, out_channels=128 * 2, kernel_size=3, padding=1) # 8*8
        self.conv6 = nn.Conv2d(in_channels=128 * 2, out_channels=128 * 2, kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(in_channels=128 * 2, out_channels=256 * 2, kernel_size=3, padding=1)
        # ]

        self.bn1 = nn.BatchNorm2d(32 * 2)
        self.bn2 = nn.BatchNorm2d(128 * 2)
        self.bn3 = nn.BatchNorm2d(256)

        self.pool = nn.MaxPool2d(2,2)
        self.dropout = nn.Dropout(0.2)

        self.fc1_mode_1 = nn.Linear(32768,256)
        self.fc1_mode_2 = nn.Linear(16384, 256)
        self.fc2_mode_3 = nn.Linear(8192,256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

        self.ReLU = nn.ReLU(inplace=True)
        self.softmax = nn.Softmax(dim=1)

    # run through prediction
    def forward(self, X):

        # run through the layers
        X = self.ReLU(self.bn1(self.conv1(X)))
        X = self.ReLU(self.conv2(X))
        X = self.ReLU(self.conv3(X))
        X = self.pool(X)

        if self.mode > 1:
            X = self.ReLU(self.bn2(self.conv4(X)))
            X = self.ReLU(self.conv5(X))
            X = self.pool(X)
            X = self.dropout(X)

        if self.mode > 2:
            X = self.ReLU(self.bn3(self.conv6(X)))
            X = self.ReLU(self.conv7(X))
            X = self.pool(X)
            X = self.dropout(X)

        # flatten
        X = X.view(X.size(0), -1)

        if self.mode == 1:
            X = self.ReLU(self.fc1_mode_1(X))
        elif self.mode == 2:
            X = self.ReLU(self.fc1_mode_2(X))
        else:
            X = self.ReLU(self.fc2_mode_3(X))

        X = self.ReLU(self.fc2(X))
        X = self.dropout(X)
        X = self.fc3(X)

        return self.softmax(X)