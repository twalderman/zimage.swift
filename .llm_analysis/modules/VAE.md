# Module: VAE (`Sources/ZImage/Model/VAE`)

## Purpose
Implements the Variational Autoencoder (VAE) used to compress images into latents (encoding) and reconstruct images from latents (decoding).

## Key Components

### `AutoencoderKL`
- **Responsibility**: Full VAE implementation (Encoder + Decoder).
- **Architecture**:
  - **Encoder**: Compresses `(B, C, H, W)` image into `(B, 2*latent_channels, h, w)` distribution parameters (mean, logvar).
  - **Decoder**: Reconstructs `(B, C, H, W)` image from `(B, latent_channels, h, w)` latents.
  - **Blocks**: Uses `VAEResnetBlock2D` and `VAEDownSampler`/`VAEUpSampler`.
  - **MidBlock**: A bottleneck block containing `VAESelfAttention` for global context.
- **Normalization**: `GroupNorm`.

### `AutoencoderDecoderOnly`
- **Responsibility**: A lightweight wrapper containing *only* the `VAEDecoder`.
- **Use Case**: Inference-only scenarios (Text-to-Image) where the Encoder is not needed, saving significant memory (VRAM/RAM).

### `VAEConfig`
- **Data**: Configuration for channel widths, block counts, and scaling factors.
- **Defaults**: Sourced from `ZImageModelMetadata`.

## Data Flow
- **Decoding**:
  1. Input Latents `(B, C, h, w)`.
  2. Transpose to `(B, h, w, C)` for MLX.
  3. Apply `scalingFactor` and `shiftFactor` (Inverse normalization).
  4. Pass through `VAEDecoder`.
  5. Transpose back to `(B, C, H, W)`.

## Implementation Details
- **Channel Ordering**: Handles the impedance mismatch between PyTorch weights (Channel First) and MLX operations (Channel Last) via transpositions.
- **MLXNN**: Uses `MLXNN.Upsample` with nearest neighbor interpolation.

## Code Quality Observations

### Sources/ZImage/Model/VAE/AutoencoderKL.swift
- **Purpose**: Variational Autoencoder (VAE) for latent-pixel conversion.
- **Components**:
  - `VAEEncoder`: Compresses image to latents.
  - `VAEDecoder`: Reconstructs image from latents.
  - `AutoencoderDecoderOnly`: Optimization for inference pipelines.
- **Observations**:
  - Standard ResNet+Attention VAE architecture.
  - Clean modular structure (UpBlock, DownBlock, MidBlock).