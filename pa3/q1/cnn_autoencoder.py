"""
Question 1: Autoencoder for MNIST (CNN)
Encoder:
    two concolution layers,
    two max-pooling layers (followed each by a convolution layers),
    Kernel Size: 3x3
    Activation: ReLU
    Padding: 1

Decoder:
    3 convolution layers
    Kernel: 3x3
    Padding: 1

Total Epochs: 10
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

sys.path.append(os.path.dirname(__file__)) 

#============ HELPERS ===============#
import helper_functions as util

class CNNAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        #encoder 
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # 1*28*28-->32*28*28
            nn.ReLU(inplace=True),                       # ReLU activation 
            nn.MaxPool2d(2),                             # 28*28-->14*14
            nn.Conv2d(32, 64, kernel_size=3, padding=1), # 32*14*14-->64*14*14
            nn.ReLU(inplace=True),                       # ReLU activation
            nn.MaxPool2d(2),                             # 14*14-->7*7
        )
        #decoder
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, 
                        mode="bilinear", 
                        align_corners=False),  

            nn.Conv2d(64, 64, 
                      kernel_size=3, 
                      padding=1),                       

            nn.LeakyReLU(0.1, 
                         inplace=True),                                  

            nn.Upsample(scale_factor=2, 
                        mode="bilinear", 
                        align_corners=False),

            nn.Conv2d(64, 32, 
                      kernel_size=3, 
                      padding=1), 

            nn.LeakyReLU(0.1, 
                         inplace=True),

            nn.Conv2d(32, 1, 
                      kernel_size=3, 
                      padding=1),
            nn.Sigmoid(),                             
        )

        # Initialize weights for all convolutional layers====#
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, nonlinearity="leaky_relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

        # Find the last Conv2d layer in the decoder
        last_conv = None
        for layer in self.decoder:
            if isinstance(layer, nn.Conv2d):
                last_conv = layer

        # Set bias of the last Conv2d to 0.5 if it exists
        if last_conv is not None and last_conv.bias is not None:
            nn.init.constant_(last_conv.bias, 0.5)
        # ====================================================#

    def forward(self, x):
        encoded = self.encoder(x) 
        return self.decoder(encoded) 

# training 
def train(model, train_loader, device, epochs=10, lr=1e-4):
    model.to(device) 

    # init our Adam optimizer 
    opt = optim.Adam(
        model.parameters(), 
        lr=lr, 
        weight_decay=1e-5) 

    # Mean Squared Error as loss function
    criterion = nn.MSELoss() 

    # optimize and train for each epoch
    for curr_epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0

        # Iterate over batches
        for x, _ in train_loader:
            x = x.to(device) # move batch to GPU 
            opt.zero_grad() # reset gradients 
            y = model(x) # reconstruct the image (forward pass) 
            loss = criterion(y, x) # compute loss 
            loss.backward() # backpropogate for optimization
            opt.step() # update parameters
            total_loss += loss.item() * x.size(0) 

        # Compute average MSE across the full dataset
        MSE = total_loss / len(train_loader.dataset)
        print(f"CNN: Epoch-{curr_epoch} | MSE: {MSE:.6f}")

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load MNIST dataset
    train_loader, test_loader = util.get_mnist_loaders(batch_size=256)

    # create model instance
    model = CNNAutoencoder()

    # print params
    params_total, params_trainable = util.count_params(model)
    encoder_params, decoder_params = util.count_params_by(model)
    print(f"Total params: {params_total}\nTrainable: {params_trainable}")
    print(f"Encoder params: {encoder_params}\nDecoder params: {decoder_params}")

    # train the model for 10 epochs
    train(model, train_loader, device, epochs=10, lr=1e-4)

    # after training, visualize a few reconstructions on test data
    out_path = Path("outputs/cnn_reconstructions.png")
    util.show_reconstruction(
        model, 
        test_loader, 
        device, 
        out_path)
    print(f"Saved reconstruction grid to: {out_path}")

if __name__ == "__main__":
    print("Running CNN autoencoder")
    main()

