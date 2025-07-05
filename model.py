# =============================================================================
# DIFFUSION MODEL ARCHITECTURE
# =============================================================================
# This file implements a U-Net based diffusion model for sprite generation.
# The model uses residual connections, time embeddings, and context embeddings
# for conditional generation.

import torch
import torch.nn as nn

class ResidualConvBlock(nn.Module):
    """
    Residual convolutional block with skip connections.
    
    This block implements a residual connection where the input is added to the
    output of two consecutive convolutions. This helps with gradient flow and
    allows the network to learn residual functions.
    
    Architecture:
        Input -> Conv1 -> BN -> GELU -> Conv2 -> BN -> GELU -> Add(Input) -> Output
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        # Check if we need to handle different channel dimensions
        self.same_channels = self.in_channels == self.out_channels
        
        # First convolution block: in_channels -> out_channels
        self.conv1 = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.out_channels),  # Normalize activations
            nn.GELU()  # Gaussian Error Linear Unit activation
        )

        # Second convolution block: out_channels -> out_channels
        self.conv2 = nn.Sequential(
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.out_channels),
            nn.GELU()
        )

    def forward(self, x):
        # Apply first convolution
        x1 = self.conv1(x)     
        # Apply second convolution
        x2 = self.conv2(x1) 
        
        # Handle residual connection
        if self.same_channels:
            # If input and output have same channels, add directly
            out = x + x2 
        else:
            # If different channels, use 1x1 conv to match dimensions
            resize = nn.Conv2d(x.shape[1], x2.shape[1], kernel_size=1, stride=1, padding=0).to(x.device)
            return resize(x) + x2
        
        # Scale down by sqrt(2) for numerical stability
        return out / 1.414     

class UnetDown(nn.Module):
    """
    Downsampling block for U-Net encoder.
    
    This block reduces spatial dimensions by half while increasing feature channels.
    It consists of two residual blocks followed by max pooling.
    
    Architecture:
        Input -> ResidualBlock1 -> ResidualBlock2 -> MaxPool2d -> Output
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.model = nn.Sequential(
                    ResidualConvBlock(in_channels, out_channels),  # First residual block
                    ResidualConvBlock(out_channels, out_channels),  # Second residual block
                    nn.MaxPool2d(2)  # Downsample by factor of 2
                    )
    def forward(self, x):
        return self.model(x)
    
class UnetUp(nn.Module):
    """
    Upsampling block for U-Net decoder.
    
    This block increases spatial dimensions by half and concatenates with skip
    connections from the encoder. It consists of transposed convolution followed
    by two residual blocks.
    
    Architecture:
        Input -> ConvTranspose2d -> Concat(Skip) -> ResidualBlock1 -> ResidualBlock2 -> Output
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.model = nn.Sequential(
                    nn.ConvTranspose2d(in_channels, out_channels, 2, 2),  # Upsample by factor of 2
                    ResidualConvBlock(out_channels, out_channels),  # First residual block
                    ResidualConvBlock(out_channels, out_channels)   # Second residual block
                    )
    def forward(self, x, skip):
        # Concatenate with skip connection from encoder
        x = torch.cat((x, skip), 1)
        return self.model(x)
    
     
class EmbedFC(nn.Module):
    """
    Fully connected embedding layer for time and context information.
    
    This module projects scalar inputs (time steps or context vectors) into
    higher dimensional embeddings that can be used throughout the network.
    
    Architecture:
        Input -> Linear -> GELU -> Linear -> Output
    """
    def __init__(self, input_dim, emb_dim):
        super().__init__()
        
        self.input_dim = input_dim
        self.emb_dim = emb_dim
        
        self.model = nn.Sequential(
                    nn.Linear(input_dim, emb_dim),  # First projection
                    nn.GELU(),  # Activation
                    nn.Linear(emb_dim, emb_dim)     # Second projection
                    )
    def forward(self, x):
        # Reshape input to 2D for linear layers
        x = x.view(-1, self.input_dim)
        return self.model(x)

    
class DiffusionModel(nn.Module):
    """
    Main diffusion model using U-Net architecture with time and context conditioning.
    
    This is the core model that learns to denoise images step by step. It takes
    a noisy image, timestep, and optional context vector as input and predicts
    the noise that was added to the original image.
    
    Architecture Overview:
    1. Encoder: Downsample and extract features
    2. Bottleneck: Process global information with time/context embeddings
    3. Decoder: Upsample and reconstruct with skip connections
    4. Output: Predict noise at each pixel
    
    Conditioning:
    - Time embedding: Tells the model which timestep it's denoising
    - Context embedding: Provides additional information (e.g., sprite attributes)
    """
    def __init__(self, in_channels, n_features, n_cfeatures, height):
        super().__init__()
        
        self.in_channels = in_channels      # Number of input channels (3 for RGB)
        self.n_features = n_features        # Base number of features (64)
        self.n_cfeatures = n_cfeatures      # Context vector dimension (5)
        self.height = height                # Image height (16)
        
        # Initial convolution to process input
        self.init_conv = ResidualConvBlock(in_channels, n_features)
        
        # Encoder: Downsampling blocks
        self.down1 = UnetDown(n_features, n_features)        # 16x16 -> 8x8
        self.down2 = UnetDown(n_features, 2 * n_features)    # 8x8 -> 4x4
        
        # Bottleneck: Global feature processing
        self.to_vec = nn.Sequential(
            nn.AvgPool2d(4),  # Global average pooling to get single feature vector
            nn.GELU()
        )
        
        # Time embeddings for different scales
        self.time_emb1 = EmbedFC(1, 2 * n_features)  # For deeper layers
        self.time_emb2 = EmbedFC(1, n_features)      # For shallower layers
        
        # Context embeddings for different scales
        self.context_emb1 = EmbedFC(n_cfeatures, 2 * n_features)  # For deeper layers
        self.context_emb2 = EmbedFC(n_cfeatures, n_features)      # For shallower layers
        
        # Decoder: Upsampling blocks
        # Special upsampling from bottleneck to decoder
        self.up0 = nn.Sequential(
                    nn.ConvTranspose2d(2 * n_features, 2 * n_features, height//4, height//4),
                    nn.GroupNorm(8, 2 * n_features),  # Group normalization for stability
                    nn.ReLU()
                    )
        
        # Standard upsampling blocks with skip connections
        self.up1 = UnetUp(4 * n_features, n_features)  # 4x4 -> 8x8
        self.up2 = UnetUp(2 * n_features, n_features)  # 8x8 -> 16x16
        
        # Final output layer
        self.out = nn.Sequential(
                    nn.Conv2d(2 * n_features, n_features, kernel_size=3, stride=1, padding=1),
                    nn.GroupNorm(8, n_features),
                    nn.ReLU(),
                    nn.Conv2d(n_features, self.in_channels, kernel_size=3, stride=1, padding=1)  # Predict noise
                    )
        
    def forward(self, x, t, c=None):
        """
        Forward pass through the diffusion model.
        
        Args:
            x (torch.Tensor): Noisy input image [B, C, H, W]
            t (torch.Tensor): Timestep tensor [B, 1, 1, 1] (normalized to [0, 1])
            c (torch.Tensor, optional): Context vector [B, n_cfeatures]
            
        Returns:
            torch.Tensor: Predicted noise [B, C, H, W]
        """
        # Encoder path
        x = self.init_conv(x)           # Initial processing
        down1 = self.down1(x)           # First downsampling
        down2 = self.down2(down1)       # Second downsampling
        
        # Bottleneck: Convert to global feature vector
        vec = self.to_vec(down2)
        
        # Handle context (use zeros if not provided)
        if c is None:
            c = torch.zeros(x.shape[0], self.n_cfeatures).to(x)

        # Create embeddings for time and context
        # Reshape to [B, C, 1, 1] for broadcasting
        cemb1 = self.context_emb1(c).view(-1, self.n_features * 2, 1, 1)    
        temb1 = self.time_emb1(t).view(-1, self.n_features * 2, 1, 1)
        cemb2 = self.context_emb2(c).view(-1, self.n_features, 1, 1)
        temb2 = self.time_emb2(t).view(-1, self.n_features, 1, 1)
        
        # Decoder path with conditioning
        up1 = self.up0(vec)  # Special upsampling from bottleneck
        # Apply conditioning and skip connections
        up2 = self.up1(up1 * cemb1 + temb1, down2)  # Deeper conditioning
        up3 = self.up2(up2 * cemb2 + temb2, down1)  # Shallower conditioning
        
        # Final output with residual connection from input
        out = self.out(torch.cat((up3, x), 1))
        
        return out
        
              