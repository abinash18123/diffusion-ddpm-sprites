# Diffusion Model for Sprite Generation

A simple PyTorch implementation of DDPM for generating 16x16 pixel sprites.

## Quick Start

```bash
# Install dependencies
pip install torch torchvision numpy matplotlib tqdm

# Train the model
python train.py

# Run inference
inference_diffusion.ipynb

## Model

- 16x16 RGB sprites
- 500 diffusion timesteps
- U-Net with residual connections
- Conditional generation support

## Model Architecture

- **Input**: 16x16 RGB images
- **Features**: 64 base features
- **Context**: 5-dimensional context vector
- **Timesteps**: 500 diffusion steps
- **Noise Schedule**: Linear from β₁=1e-4 to β₂=0.02

## Project Structure

  diffusion(DDPM)/
  ├── model.py              # U-Net diffusion model
  ├── train.py              # Training script
  ├── utilities.py          # Dataset and inference utilities
  ├── inference_diffusion.ipynb  # Interactive notebook
  ├── dataset/              # Sprite dataset
  └── weights/              # Model checkpoints


## References

- [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) - Ho et al.
- [U-Net Architecture](https://arxiv.org/abs/1505.04597) - Ronneberger et al.

