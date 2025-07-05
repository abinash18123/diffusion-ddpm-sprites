
# This script implements distributed training for the DDPM diffusion model.
# It uses PyTorch's DistributedDataParallel (DDP) for multi-GPU training
# and implements the standard DDPM training procedure.

import os
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from model import *
from utilities import *


# Diffusion process parameters
timesteps = 500        # Number of diffusion timesteps (T)
beta1 = 1e-4          # Initial noise level (β₁)
beta2 = 0.02          # Final noise level (β₂)

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else torch.device('cpu'))

# Model architecture parameters
n_features = 64       # Base number of features in the model
n_cfeatures = 5       # Dimension of context vector
height = 16           # Image height (16x16 sprites)

# Training parameters
save_dir = 'weights/'  # Directory to save model checkpoints
batch_size = 100      # Batch size per GPU
n_epoch = 320         # Total number of training epochs
lrate = 1e-3          # Initial learning rate


# Construct the DDPM noise schedule
# This creates a linear schedule from β₁ to β₂ over T timesteps
b_t = (beta2 - beta1) * torch.linspace(0, 1, timesteps + 1, device=device) + beta1
a_t = 1 - b_t  # α_t = 1 - β_t
ab_t = torch.cumsum(a_t.log(), dim=0).exp()  # ᾱ_t = ∏(1 - β_i) from i=1 to t
ab_t[0] = 1  # Set ᾱ_0 = 1 for consistency

def perturb_input(x, t, noise, beta1=1e-4, beta2=0.02, timesteps=1000):
    """
    Add noise to input images according to the diffusion process.
    
    This function implements the forward diffusion process:
    q(x_t | x_0) = N(x_t; √ᾱ_t * x_0, (1 - ᾱ_t) * I)
    
    Args:
        x (torch.Tensor): Original images [B, C, H, W]
        t (torch.Tensor): Timestep indices [B]
        noise (torch.Tensor): Random noise to add [B, C, H, W]
        beta1, beta2 (float): Noise schedule parameters
        timesteps (int): Total number of timesteps
        
    Returns:
        torch.Tensor: Noisy images at timestep t
    """
    device = x.device  # Ensure everything is on the correct GPU
    
    # Recompute noise schedule for the given parameters
    b_t = (beta2 - beta1) * torch.linspace(0, 1, timesteps + 1, device=device) + beta1
    a_t = 1 - b_t
    ab_t = torch.cumsum(a_t.log(), dim=0).exp()
    ab_t[0] = 1.0

    ab_t = ab_t.to(device)  # Move to correct device

    # Apply forward diffusion: x_t = √ᾱ_t * x_0 + √(1 - ᾱ_t) * ε
    return ab_t.sqrt()[t, None, None, None] * x + (1 - ab_t[t, None, None, None]) * noise

def train(rank, world_size):
    """
    Training function for a single process in distributed training.
    
    This function sets up the distributed environment, initializes the model,
    and runs the training loop for the specified number of epochs.
    
    Args:
        rank (int): Process rank (0 to world_size-1)
        world_size (int): Total number of processes
    """
    # DISTRIBUTED TRAINING SETUP
    # Initialize distributed training environment
    os.environ["MASTER_ADDR"] = "localhost"  # Master node address
    os.environ["MASTER_PORT"] = "12355"      # Master node port
    
    # Initialize process group for distributed training
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)  # Set GPU for this process

    # MODEL INITIALIZATION
    # Create model and wrap with DDP for distributed training
    model = DiffusionModel(
        in_channels=3, 
        n_features=n_features, 
        n_cfeatures=n_cfeatures, 
        height=height
    ).to(rank)
    model = DDP(model, device_ids=[rank])  # Wrap with DDP

    # Initialize optimizer
    optim = torch.optim.Adam(model.parameters(), lr=lrate)

    # DATASET AND DATALOADER SETUP
    # Load dataset
    dataset = CustomDataset(
        "dataset/sprites_1788_16x16.npy", 
        "dataset/sprite_labels_nc_1788_16x16.npy", 
        transform
    )
    
    # Create distributed sampler for proper data distribution across GPUs
    sampler = DistributedSampler(
        dataset, 
        num_replicas=world_size, 
        rank=rank, 
        shuffle=True
    )
    
    # Create dataloader with distributed sampler
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        sampler=sampler, 
        num_workers=4
    )

    # TRAINING LOOP
    for ep in range(n_epoch):
        # Set epoch for sampler to ensure different shuffling each epoch
        sampler.set_epoch(ep)
        print(f"Rank {rank} - Epoch {ep}")

        # Linear learning rate decay
        optim.param_groups[0]['lr'] = lrate * (1 - ep / n_epoch)

        # Progress bar (only show for rank 0)
        pbar = tqdm(dataloader, mininterval=2, disable=(rank != 0))
        
        for x, _ in pbar:
            # FORWARD PASS

            optim.zero_grad()  # Clear gradients
            x = x.to(rank)     # Move data to GPU

            # Generate random noise for the diffusion process
            noise = torch.randn_like(x)
            
            # Sample random timesteps for each image in the batch
            t = torch.randint(1, timesteps + 1, (x.shape[0],)).to(rank)
            
            # Add noise to images according to the diffusion process
            x_pert = perturb_input(x, t, noise)

            # Predict the noise using the model
            pred_noise = model(x_pert, t / timesteps)

            # LOSS COMPUTATION AND BACKWARD PASS
            # Compute MSE loss between predicted and actual noise
            loss = F.mse_loss(pred_noise, noise)
            
            # Backward pass
            loss.backward()
            optim.step()

        # Save model checkpoint at the end of training (only rank 0)
        if rank == 0 and (ep == n_epoch - 1):
            os.makedirs(save_dir, exist_ok=True)
            torch.save(
                model.module.state_dict(), 
                os.path.join(save_dir, f"model_{ep}.pth")
            )
            print(f"Saved model at epoch {ep}")

    # Clean up distributed training
    dist.destroy_process_group()

def main():
    """
    Main function to launch distributed training.
    
    This function determines the number of available GPUs and spawns
    training processes for each GPU.
    """
    # Get number of available GPUs
    world_size = torch.cuda.device_count()
    
    # Spawn training processes (one per GPU)
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()