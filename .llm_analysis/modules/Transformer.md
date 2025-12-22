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
