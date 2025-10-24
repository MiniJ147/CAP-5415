from __future__ import print_function
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
from ConvNet import ConvNet 
import argparse
import numpy as np 
import matplotlib.pyplot as plt

def is_correct(pred, ground_truth):
    total = 0
    for idx in range(len(pred)):
        total += 1 if pred[idx, 0].item() == ground_truth[idx] else 0
    return total

def train(model, device, train_loader, optimizer, criterion, epoch, batch_size):
    '''
    Trains the model for an epoch and optimizes it.
    model: The model to train. Should already be in correct device.
    device: 'cuda' or 'cpu'.
    train_loader: dataloader for training samples.
    optimizer: optimizer to use for model parameter updates.
    criterion: used to compute loss for prediction and target 
    epoch: Current epoch to train for.
    batch_size: Batch size to be used.
    '''
    
    # Set model to train mode before each epoch
    model.train()
    
    # Empty list to store losses 
    losses = []
    correct = 0
    
    # Iterate over entire training samples (1 epoch)
    for batch_idx, batch_sample in enumerate(train_loader):
        data, target = batch_sample
        ground_truth = batch_sample[1]

        # Push data/label to correct device
        data, target = data.to(device), target.to(device)
        
        # Reset optimizer gradients. Avoids grad accumulation (accumulation used in RNN).
        optimizer.zero_grad()
        
        # Do forward pass for current set of data
        output = model(data)
        
        # ======================================================================
        # Compute loss based on criterion
        # ----------------- YOUR CODE HERE ----------------------
        loss = criterion(output, target) 
        
        # Computes gradient based on final loss
        loss.backward()
        
        # Store loss
        losses.append(loss.item())
        
        # Optimize model parameters based on learning rate and gradient 
        optimizer.step()
        
        # Get predicted index by selecting maximum log-probability
        pred = output.argmax(dim=1, keepdim=True)
        
        # ======================================================================
        # Count correct predictions overall 
        # ----------------- YOUR CODE HERE ---------------------- 
        correct += is_correct(pred=pred, ground_truth=ground_truth)

    train_loss = float(np.mean(losses))
    train_acc = correct / ((batch_idx+1) * batch_size)
    print('Train set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        float(np.mean(losses)), correct, (batch_idx+1) * batch_size,
        100. * correct / ((batch_idx+1) * batch_size)))
    return train_loss, train_acc
    


def test(model, device, test_loader):
    '''
    Tests the model.
    model: The model to train. Should already be in correct device.
    device: 'cuda' or 'cpu'.
    test_loader: dataloader for test samples.
    '''
    
    # Set model to eval mode to notify all layers.
    model.eval()
    
    losses = []
    correct = 0
    
    # Set torch.no_grad() to disable gradient computation and backpropagation
    with torch.no_grad():
        for batch_idx, sample in enumerate(test_loader):
            data, target = sample
            ground_truth = sample[1]
            data, target = data.to(device), target.to(device)
            

            # Predict for data by doing forward pass
            output = model(data)
            
            # ======================================================================
            # Compute loss based on same criterion as training
            # ----------------- YOUR CODE HERE ----------------------
            loss = nn.CrossEntropyLoss()(output, target)
            
            # Append loss to overall test loss
            losses.append(loss.item())
            
            # Get predicted index by selecting maximum log-probability
            pred = output.argmax(dim=1, keepdim=True)
            
            # ======================================================================
            # Count correct predictions overall 
            # ----------------- YOUR CODE HERE ----------------------
            correct += is_correct(pred=pred, ground_truth=ground_truth)

    test_loss = float(np.mean(losses))
    accuracy = 100. * correct / len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), accuracy))
    
    return test_loss, accuracy
    

def run_main(FLAGS):
    # Check if cuda is available
    use_cuda = torch.cuda.is_available()
    
    # Set proper device based on cuda availability 
    device = torch.device("cuda" if use_cuda else "cpu")
    print("Torch device selected: ", device)
    
    # Initialize the model and send to device 
    model = ConvNet(FLAGS.mode).to(device)
    
    # ======================================================================
    # Define loss function.
    # ----------------- YOUR CODE HERE ----------------------
    criterion = nn.CrossEntropyLoss()
    
    # ======================================================================
    # Define optimizer function.
    # ----------------- YOUR CODE HERE ----------------------
    # optimizer = optim.Adam(model.parameters(), FLAGS.learning_rate)
    optimizer = optim.SGD(model.parameters(), lr=FLAGS.learning_rate)
        
    
    # Create transformations to apply to each data sample 
    # Can specify variations such as image flip, color flip, random crop, ...
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    
    # Load datasets for training and testing
    # Inbuilt datasets available in torchvision (check documentation online)
    dataset1 = datasets.MNIST('./data/', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.MNIST('./data/', train=False,
                       transform=transform)
    train_loader = DataLoader(dataset1, batch_size = FLAGS.batch_size, 
                                shuffle=True, num_workers=4)
    test_loader = DataLoader(dataset2, batch_size = FLAGS.batch_size, 
                                shuffle=False, num_workers=4)
    
    best_accuracy = 0.0
    
    domain = np.arange(1,FLAGS.num_epochs+1)
    train_loss_y = np.array([])
    test_loss_y = np.array([])
    train_accuracy_y = np.array([])
    test_accuracy_y = np.array([])

    # Run training for n_epochs specified in config 
    for epoch in range(1, FLAGS.num_epochs + 1):
        print("Epoch-Iter:", epoch, FLAGS.num_epochs)
        train_loss, train_accuracy = train(model, device, train_loader,
                                            optimizer, criterion, epoch, FLAGS.batch_size)
        test_loss, test_accuracy = test(model, device, test_loader)

        train_loss_y = np.append(train_loss_y, train_loss)
        test_loss_y = np.append(test_loss_y, test_loss)

        train_accuracy_y = np.append(train_accuracy_y, train_accuracy * 100)
        test_accuracy_y = np.append(test_accuracy_y, test_accuracy)
        
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
    
     
    print("accuracy is {:2.2f}".format(best_accuracy)) 
    print("Training and evaluation finished")

    # plotting loss vs epoch
    plt.plot(domain, train_loss_y, color='blue', label="train_loss")
    plt.plot(domain, test_loss_y, color='red',label='test_loss')
    plt.title(f"Loss vs Epoch (Mode: {FLAGS.mode})")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.ylim(bottom=0)
    plt.savefig(f"Loss_vs_Epoch_Mode_{FLAGS.mode}")
    plt.close()

    # plotting accuracy vs epoch
    plt.plot(domain, train_accuracy_y, color='blue',label='train_accuracy')
    plt.plot(domain, test_accuracy_y, color='red',label='test_accuracy')
    plt.title(f"Accuracy vs Epoch (Mode: {FLAGS.mode})")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.ylim(bottom=0)
    plt.savefig(f"Accuracy_vs_Epoch_Mode_{FLAGS.mode}")
    plt.close()

    
    
if __name__ == '__main__':
    # Set parameters for Sparse Autoencoder
    parser = argparse.ArgumentParser('CNN Exercise.')
    parser.add_argument('--mode',
                        type=int, default=1,
                        help='Select mode between 1-5.')
    parser.add_argument('--learning_rate',
                        type=float, default=0.1,
                        help='Initial learning rate.')
    parser.add_argument('--num_epochs',
                        type=int,
                        default=60,
                        help='Number of epochs to run trainer.')
    parser.add_argument('--batch_size',
                        type=int, default=10,
                        help='Batch size. Must divide evenly into the dataset sizes.')
    parser.add_argument('--log_dir',
                        type=str,
                        default='logs',
                        help='Directory to put logging.')
    
    FLAGS = None
    FLAGS, unparsed = parser.parse_known_args()
    
    run_main(FLAGS)
    
    