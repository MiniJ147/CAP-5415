"""
The purpose of this file is to provide helper functions used by both models.
"""

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def get_mnist_loaders(batch_size = 256, num_workers = 0):
    """
    returns train/test dataLoaders for MNIST.
    downloads / and returns dataloaders for the MNIST dataset.
    return train_loader, test_loader
    """
    tfm = transforms.ToTensor()

    # MNIST DATASETS
    train_ds = datasets.MNIST(
        root="./data", 
        train=True, 
        download=True, 
        transform=tfm)
    test_ds  = datasets.MNIST(
        root="./data", 
        train=False, 
        download=True, 
        transform=tfm)

    # Data Loaders
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,  
        pin_memory=False,        
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )

    # return loaders
    return train_loader, test_loader

def count_params(model: torch.nn.Module):
    """
    returns the count of total params and trainble params
    return total, trainable
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def count_params_by(module: torch.nn.Module):
    """
    returns total number of params given the model. 
    return encoder_params, decoder_params
    """
    encoder_params = sum(p.numel() for p in module.encoder.parameters())
    decoder_params = sum(p.numel() for p in module.decoder.parameters())
    return encoder_params, decoder_params

@torch.no_grad()
def show_reconstruction(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    output_path,
    per_class: int = 2,
):
    """
    generate and save a grid comparing input and reconstructed images.
    """
    import matplotlib.pyplot as plt
    from pathlib import Path

    # Ensure model is on correct device and in eval mode
    model.eval()
    model.to(device)

    # Prepare directories
    dest = Path(output_path)
    dest.parent.mkdir(parents=True, exist_ok=True)

    # Initialize class image collectors
    samples = {k: [] for k in range(10)}

    # Gather examples
    for images, labels in loader:
        images = images.to(device)
        for img, lbl in zip(images, labels):
            lbl = int(lbl)
            if len(samples[lbl]) < per_class:
                samples[lbl].append(img.unsqueeze(0))
        if all(len(v) == per_class for v in samples.values()):
            break

    # Define figure grid size
    total_cols = 10 * per_class
    fig, axs = plt.subplots(2, total_cols, figsize=(total_cols, 4))

    # Fill grid with original and reconstructed images
    idx = 0
    for _, imgs in samples.items():
        for img in imgs:
            original = img.cpu().squeeze().numpy()
            reconstructed = model(img).clamp(0, 1).cpu().squeeze().numpy()

            for row, data in enumerate((original, reconstructed)):
                axs[row, idx].imshow(data, cmap="gray")
                axs[row, idx].axis("off")

            idx += 1

    # Save and clean up
    fig.tight_layout(pad=0.2)
    fig.savefig(dest, bbox_inches="tight")
    plt.close(fig)

