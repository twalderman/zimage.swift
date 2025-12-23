# Module: Transformer (DiT) (`Sources/ZImage/Model/Transformer`)

## Purpose
This is the core generative engine of the project. It implements a Diffusion Transformer (DiT) that operates on latent image patches and text embeddings to denoise the image.

## Key Components

### `ZImageTransformer2DModel`
- **Responsibility**: The top-level model container.
- **Architecture**:
  1.  **Embedders**:
      - `tEmbedder`: Timestep embedding.
      - `allXEmbedder`: Image patch embedding (Patch size 2x2).
      - `capEmbedder`: Caption/Text embedding projection.
      - `ropeEmbedder`: Rotational Positional Embedding generator.
  2.  **Refiners (Dual-Stream Pre-Processing)**:
      - `noiseRefiner`: Processes the image latent stream independently.
      - `contextRefiner`: Processes the caption embedding stream independently.
  3.  **Main Layers (Joint Processing)**:
      - `layers`: Concatenates the refined image and caption streams into a single sequence and processes them jointly with self-attention.
  4.  **Final Layer**: Projects the processed image tokens back to latent space.
- **Caching**: Uses `TransformerCache` to pre-compute RoPE frequencies (`freqsCis`) and masks for efficient variable-resolution inference.

### `ZImageTransformerBlock`
- **Responsibility**: A single transformer layer.
- **Features**:
  - **AdaLN Modulation**: Uses Adaptive Layer Normalization conditioned on the timestep embedding (`adalnInput`). It computes scale and shift parameters (`attnScale`, `attnGate`, etc.) to modulate the normalization layers.
  - **RMSNorm**: Standard pre-normalization.
  - **FeedForward**: SwiGLU-style MLP (inferred from `ZImageFeedForward` usage).

### `ZImageSelfAttention`
- **Responsibility**: Multi-head self-attention mechanism.
- **Features**:
  - **RoPE**: Applies Rotary Positional Embeddings to Queries and Keys.
  - **QK Norm**: Optional normalization of Q and K before attention (stabilizes training).
  - **MLXFast**: Uses `MLXFast.scaledDotProductAttention` for performance.

## Data Flow
1.  **Inputs**: Latents, Timestep, Text Embeddings.
2.  **Embedding**: Latents -> Patches; Timestep -> Vector; Text -> Projected.
3.  **Refinement**: Image and Text streams processed separately by `noiseRefiner` and `contextRefiner`.
4.  **Unification**: Streams concatenated.
5.  **Joint Processing**: Deep stack of `layers` processes the unified stream.
6.  **Output**: Image tokens extracted, unpatched, and projected to output latents.

## Unique Characteristics
- **Hybrid Architecture**: Starts with independent streams (like a dual-encoder) and merges into a single stream (like a concat-transformer). This likely balances computational efficiency with cross-modal interaction.
- **Complex RoPE**: Handles 2D (spatial) and potentially 3D (temporal/video) positional embeddings via `axesDims` and `axesLens`.

## Code Quality Observations

### Sources/ZImage/Model/Transformer/ZImageTransformer2D.swift
- **Purpose**: Core Diffusion Transformer (DiT) model.
- **Architecture**:
  - `TimeEmbedder` -> `PatchEmbedder`
  - `NoiseRefiner` (Blocks) -> `ContextRefiner` (Blocks) -> `Main Layers` (Blocks)
  - `FinalLayer` -> Unpatch
- **Observations**:
  - **Caching**: Uses `TransformerCache` for efficient RoPE computation.
  - **Patching**: Hardcoded patch size (2x2) and frame patch size (1).
  - **Coupling**: Includes weight loading logic directly in the model class.

### Sources/ZImage/Model/Transformer/ZImageControlTransformer2D.swift
- **Purpose**: Modified DiT for ControlNet support.
- **Observations**:
  - **High Duplication**: Reimplements most of `ZImageTransformer2D` to inject control signals (hints).
  - **Control Logic**: Adds `controlLayers` and `controlNoiseRefiner`.
  - **Forward Pass**: Complex coordination of main stream and control stream.

### Sources/ZImage/Model/Transformer/ZImageTransformerBlock.swift (and Control/Base variants)
- **Purpose**: Single Transformer Block (Attention + MLP + AdaLN).
- **Observations**:
  - `ZImageTransformerBlock`: Standard block.
  - `ZImageControlTransformerBlock`: Adds zero-conv projections for ControlNet.
  - `BaseZImageTransformerBlock`: Version that accepts external hints (used in Control Transformer).
  - **Refactoring Opportunity**: These three could likely be unified into a single flexible block definition.

### Sources/ZImage/Model/Transformer/ZImageSelfAttention.swift
- **Purpose**: Self-Attention mechanism.
- **Observations**:
  - Supports QK Norm (Query/Key Normalization).
  - Uses `MLXFast.scaledDotProductAttention`.
  - Integrates 3D RoPE.

### Sources/ZImage/Model/Transformer/ZImageRopeEmbedder.swift
- **Purpose**: 3D Rotary Positional Embeddings.
- **Observations**:
  - Handles 3 dimensions (Frames, Height, Width).
  - Precomputes frequency tables for efficiency.

### Sources/ZImage/Model/Transformer/TransformerCacheBuilder.swift
- **Purpose**: Helper to build caching structures for RoPE and masks.
- **Observations**:
  - Handles padding to ensure sequence lengths are multiples of 32 (likely for hardware efficiency/kernel requirements).
  - Precomputes positional grids.

### Sources/ZImage/Model/Transformer/ZImageCoordinateUtils.swift
- **Purpose**: Generates coordinate grids for RoPE.
- **Observations**:
  - Simple utility, separates concerns well.