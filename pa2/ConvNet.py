import time
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNet(nn.Module):
    def __init__(self, mode):
        super(ConvNet, self).__init__()
        
        # Define various layers here, such as in the tutorial example
        # self.conv1 = nn.Conv2D(...)
                # This will select the forward pass function based on mode for the ConvNet.
        # Based on the question, you have 5 modes available for step 1 to 5.
        # During creation of each ConvNet model, you will assign one of the valid mode.
        # This will fix the forward function (and the network graph) for the entire training/testing

        if mode == 1:
            # layers
            self.fc1 = nn.Linear(28*28,100)
            self.fc2 = nn.Linear(100,10)
            self.activation = nn.Sigmoid()
    
            self.forward = self.model_1
        elif mode == 2:
            self.conv1 = nn.Conv2d(1, 40, kernel_size=5, stride=1)
            self.conv2 = nn.Conv2d(40, 40, kernel_size=5, stride=1)
            self.pool = nn.MaxPool2d(2, 2)
            self.activation = nn.Sigmoid()
            self.fc1 = nn.Linear(40*4*4, 100)  # after conv/pool
            self.fc2 = nn.Linear(100, 10)

            self.forward = self.model_2
        elif mode == 3:
            # same as step 2 but we just change ReLU and lr=0.03
            self.conv1 = nn.Conv2d(1, 40, kernel_size=5, stride=1)
            self.conv2 = nn.Conv2d(40, 40, kernel_size=5, stride=1)
            self.pool = nn.MaxPool2d(2, 2)
            self.activation = nn.ReLU()
            self.fc1 = nn.Linear(40*4*4, 100)
            self.fc2 = nn.Linear(100, 10)

            self.forward = self.model_3
        elif mode == 4:
            # add another FC layer of 100 neurons
            self.conv1 = nn.Conv2d(1, 40, kernel_size=5, stride=1)
            self.conv2 = nn.Conv2d(40, 40, kernel_size=5, stride=1)
            self.pool = nn.MaxPool2d(2, 2)
            self.activation = nn.ReLU()
            self.fc1 = nn.Linear(40*4*4, 100)
            self.fc2 = nn.Linear(100, 100)    # new FC
            self.fc3 = nn.Linear(100, 10)

            self.forward = self.model_4
        elif mode == 5:
            # FC layers --> 1000 neurons, and Dropout 0.5
            self.conv1 = nn.Conv2d(1, 40, kernel_size=5, stride=1)
            self.conv2 = nn.Conv2d(40, 40, kernel_size=5, stride=1)
            self.pool = nn.MaxPool2d(2, 2)
            self.activation = nn.ReLU()
            self.fc1 = nn.Linear(40*4*4, 1000)
            self.fc2 = nn.Linear(1000, 1000)
            self.fc3 = nn.Linear(1000, 10)
            self.dropout = nn.Dropout(0.5)

            self.forward = self.model_5
        else: 
            print("Invalid mode ", mode, "selected. Select between 1-5")
            exit(0)
        
        
    # Baseline model. step 1
    def model_1(self, X):
        # ======================================================================
        # One fully connected layer.
        # ======================================================================
        X = X.view(X.size(0),-1)
        X = self.fc1(X) # input layer
        X = self.activation(X) # non linear
        X = self.fc2(X) # output layer
        return X

    # Use two convolutional layers.
    def model_2(self, X):
        # ======================================================================
        # Two convolutional layers + one fully connnected layer.
        #
        # ----------------- YOUR CODE HERE ----------------------
        X = self.activation(self.pool(self.conv1(X))) # layer 1 conv1 --> pool --> activation
        X = self.activation(self.pool(self.conv2(X))) # layer 2 conv2 --> pool --> activation
        X = X.view(X.size(0), -1) # flatten array
        X = self.activation(self.fc1(X)) # fc layer 1
        X = self.fc2(X) # output

        return X


    # Replace sigmoid with ReLU.
    def model_3(self, X):
        # ======================================================================
        # Two convolutional layers + one fully connected layer, with ReLU.
        #
        # ----------------- YOUR CODE HERE ----------------------
        X = self.activation(self.pool(self.conv1(X))) # layer 1 conv1 --> pool --> activation
        X = self.activation(self.pool(self.conv2(X))) # layer 2 conv2 --> pool --> activation
        X = X.view(X.size(0), -1) # flatten array
        X = self.activation(self.fc1(X)) # fc layer 1
        X = self.fc2(X) # output

        return X 

    # Add one extra fully connected layer.
    def model_4(self, X):
        # ======================================================================
        # Two convolutional layers + two fully connected layers, with ReLU.
        #
        # ----------------- YOUR CODE HERE ----------------------
        X = self.activation(self.pool(self.conv1(X))) # layer 1 conv1 --> pool --> activation
        X = self.activation(self.pool(self.conv2(X))) # layer 2 conv2 --> pool --> activation
        X = X.view(X.size(0), -1) # flatten
        X = self.activation(self.fc1(X))
        X = self.activation(self.fc2(X))
        X = self.fc3(X)
        return X

    # Use Dropout now.
    def model_5(self, X):
        # ======================================================================
        # Two convolutional layers + two fully connected layers, with ReLU.
        # and  + Dropout.
        #
        # ----------------- YOUR CODE HERE ----------------------
        X = self.activation(self.pool(self.conv1(X))) # layer 1 conv1 --> pool --> activation
        X = self.activation(self.pool(self.conv2(X))) # layer 2 conv2 --> pool --> activation
        X = X.view(X.size(0), -1) # flatten
        X = self.dropout(self.activation(self.fc1(X)))
        X = self.dropout(self.activation(self.fc2(X)))
        X = self.fc3(X)
        
        return X