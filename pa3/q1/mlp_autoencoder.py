"""
Q1 Autoencoder for MNIST (Fully Connected Layers)
Encoder: 2 Layers (256, 128) nuerons respectively
Decoder: 2 layers (256, 784) nuerons respectively
"""

# imports
from pathlib import Path

# torch imports
import torch, torch.nn as nn
import torch.optim as optim

#============ HELPERS ===============#
import helper_functions as util

class MLPAutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        #encoder
        self.encoder = nn.Sequential(
            nn.Flatten(), # converts 28*28 image into vector 
            nn.Linear(28*28, 256), 
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True))
        #decoder
        self.decoder = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 28*28),
            nn.Sigmoid())

    # model logic for each train iter
    def forward(self, x):
        encoded = self.encoder(x) # first encode
        output = self.decoder(encoded) # attempt to decode our image 
        return output.view(-1, 1, 28, 28) # return to orginal resolution

def train(model, train_loader, device, epochs=10, lr=1e-3):
    model.to(device) # attempt to use GPU

    # init our Adam optimizer
    opt = optim.Adam(model.parameters(), lr=lr)

    # Mean Squared Error as loss function 
    criterion = nn.MSELoss()

    # train model
    model.train()

    # optimize and train for each epoch 
    for curr_epoch in range(1, epochs+1):
        total_loss = 0 # stores loss per epoch

        # iterate through each batch 
        for x, _ in train_loader:
            x = x.to(device) # move batch to GPU 
            opt.zero_grad() # reset gradients 
            y = model(x) # reconstruct the image (forward pass) 
            loss = criterion(y, x) # compute loss
            loss.backward() # backpropogate for optimization
            opt.step() # update weights 
            total_loss += loss.item() * x.size(0) 
        
        MSE = total_loss / len(train_loader.dataset)
        print(f"MLP: Epoch-{curr_epoch} | MSE: {MSE:.6f}")

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load MNIST dataset
    train_loader, test_loader = util.get_mnist_loaders(batch_size=256)

    model = MLPAutoEncoder()
    
    # print params
    params_total, params_trainable = util.count_params(model)
    encoder_params, decoder_params = util.count_params_by(model)
    print(f"Total params: {params_total}\nTrainable: {params_trainable}")
    print(f"Encoder params: {encoder_params}\nDecoder params: {decoder_params}")

    train(model, train_loader, device, epochs=10, lr=1e-3)

    # Save 20 reconstructions (2 per class)
    out_path = Path('outputs/mlp_output.png')
    util.show_reconstruction(
        model, 
        test_loader, 
        device, 
        out_path)
    print(f"Saved to: {out_path}")

if __name__ == "__main__":
    print("Running MLP autoencoder")
    main()
