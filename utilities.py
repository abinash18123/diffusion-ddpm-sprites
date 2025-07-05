import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from model import DiffusionModel

# Global configuration variables
IN_CHANNELS = 3
N_FEATURES = 64
N_CFEATURES = 5
HEIGHT = 16
TIMESTEPS = 500
BETA1 = 1e-4
BETA2 = 0.02

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Global variables for noise schedule (will be set by setup_noise_schedule)
b_t = None
a_t = None
ab_t = None
model = None

#utilities
class CustomDataset(Dataset):
    def __init__(self, sfilename, lfilename, transform):
        self.sprites = np.load(sfilename)
        self.slabels = np.load(lfilename)
        print(f"sprite shape: {self.sprites.shape}")
        print(f"labels shape: {self.slabels.shape}")
        self.transform = transform
        
    def __len__(self):
        return len(self.sprites)
    
    def __getitem__(self, idx):
        image = self.transform(self.sprites[idx])
        label = torch.tensor(self.slabels[idx]).to(torch.int64)
        return (image, label)

transform = transforms.Compose([
    transforms.ToTensor(),                # from [0,255] to range [0.0,1.0]
    transforms.Normalize((0.5,), (0.5,))  # range [-1,1]
])

def load_trained_model(checkpoint_path='weights/model_63.pth'):
    """
    Load the trained diffusion model from checkpoint.
    
    Args:
        checkpoint_path (str): Path to the model checkpoint file
        
    Returns:
        DiffusionModel: Loaded model on the specified device
    """
    global model
    print(f"Loading model from: {checkpoint_path}")
    
    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    
    # Initialize model with same architecture as training
    model = DiffusionModel(
        in_channels=IN_CHANNELS,
        n_features=N_FEATURES,
        n_cfeatures=N_CFEATURES,
        height=HEIGHT
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()  # Set to evaluation mode
    
    print("Model loaded successfully!")
    return model

def setup_noise_schedule():
    """
    Construct the DDPM noise schedule.
    
    Returns:
        tuple: (b_t, a_t, ab_t) noise schedule tensors
    """
    global b_t, a_t, ab_t
    # Linear noise schedule from beta1 to beta2
    b_t = (BETA2 - BETA1) * torch.linspace(0, 1, TIMESTEPS + 1, device=device) + BETA1
    a_t = 1 - b_t
    ab_t = torch.cumsum(a_t.log(), dim=0).exp()
    ab_t[0] = 1
    
    return b_t, a_t, ab_t

def denoise_add_noise(x, t, pred_noise, z=None):
    """
    Denoise step: removes predicted noise and adds some noise back to avoid collapse.
    
    Args:
        x (torch.Tensor): Current noisy image
        t (int): Current timestep
        pred_noise (torch.Tensor): Predicted noise from the model
        z (torch.Tensor, optional): Additional noise to add back
        
    Returns:
        torch.Tensor: Denoised image with some noise added back
    """
    if z is None:
        z = torch.randn_like(x)
    
    # Calculate noise to add back
    noise = b_t.sqrt()[t] * z
    
    # Calculate denoised mean
    mean = (x - pred_noise * ((1 - a_t[t]) / (1 - ab_t[t]).sqrt())) / a_t[t].sqrt()
    
    return mean + noise

@torch.no_grad()
def sample_ddpm(n_sample=32, save_rate=20, verbose=True):
    """
    Generate samples using the standard DDPM algorithm.
    
    Args:
        n_sample (int): Number of samples to generate
        save_rate (int): How often to save intermediate results for visualization
        verbose (bool): Whether to print progress
        
    Returns:
        tuple: (final_samples, intermediate_samples)
    """
    print(f"Generating {n_sample} samples using DDPM...")
    
    # Start with pure noise: x_T ~ N(0, 1)
    samples = torch.randn(n_sample, IN_CHANNELS, HEIGHT, HEIGHT).to(device)
    
    # Array to keep track of generated steps for plotting
    intermediate = []
    
    # Reverse diffusion process: T -> 0
    for i in tqdm(range(TIMESTEPS, 0, -1), desc="Sampling", disable=not verbose):
        # Reshape time tensor for model input
        t = torch.tensor([i / TIMESTEPS])[:, None, None, None].to(device)
        
        # Sample random noise to inject back (except for final step)
        z = torch.randn_like(samples) if i > 1 else 0
        
        # Predict noise using the model
        eps = model(samples, t)
        
        # Denoise step
        samples = denoise_add_noise(samples, i, eps, z)
        
        # Save intermediate results for visualization
        if i % save_rate == 0 or i == TIMESTEPS or i < 8:
            intermediate.append(samples.detach().cpu().numpy())
    
    intermediate = np.stack(intermediate)
    print("Sampling completed!")
    
    return samples, intermediate


def visualize_samples(samples, title="Generated Samples", n_cols=8, figsize=(16, 8)):
    """
    Visualize generated samples in a grid.
    
    Args:
        samples (torch.Tensor): Generated samples
        title (str): Plot title
        n_cols (int): Number of columns in the grid
        figsize (tuple): Figure size
    """
    n_samples = len(samples)
    n_rows = (n_samples + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for i, img in enumerate(samples):
        row = i // n_cols
        col = i % n_cols
        
        # Convert from [-1, 1] to [0, 1] range and transpose for matplotlib
        image = (img.permute(1, 2, 0).clip(-1, 1).detach().cpu().numpy() + 1) / 2
        axes[row, col].imshow(image)
        axes[row, col].axis("off")
    
    # Hide empty subplots
    for i in range(n_samples, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].axis("off")
    
    plt.tight_layout()
    plt.suptitle(title, fontsize=16, y=0.98)
    plt.subplots_adjust(top=0.92)
    plt.show()

def visualize_diffusion_process(intermediate_samples, n_samples=8, n_timesteps=10):
    """
    Visualize the diffusion process showing how noise is gradually removed.
    
    Args:
        intermediate_samples (np.ndarray): Intermediate samples from sampling
        n_samples (int): Number of samples to show
        n_timesteps (int): Number of timesteps to show
    """
    # Select timesteps to show
    timestep_indices = np.linspace(0, len(intermediate_samples) - 1, n_timesteps, dtype=int)
    
    fig, axes = plt.subplots(n_samples, n_timesteps, figsize=(20, 2 * n_samples))
    
    for i in range(n_samples):
        for j, t_idx in enumerate(timestep_indices):
            img = intermediate_samples[t_idx, i]
            # Convert from [-1, 1] to [0, 1] range
            img = (img.transpose(1, 2, 0).clip(-1, 1) + 1) / 2
            
            axes[i, j].imshow(img)
            axes[i, j].axis("off")
            
            if i == 0:
                axes[i, j].set_title(f"t={t_idx}", fontsize=10)
    
    plt.tight_layout()
    plt.suptitle("Diffusion Process: Noise to Image", fontsize=16, y=0.98)
    plt.subplots_adjust(top=0.92)
    plt.show()