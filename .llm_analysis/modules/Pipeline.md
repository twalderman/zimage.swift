# Module: Pipeline & Generation (`Sources/ZImage/Pipeline`)

## Purpose
Orchestrates the end-to-end image generation process. It ties together the Model components (Transformer, TextEncoder, VAE) and the Scheduler to turn a user prompt into a final image.

## Key Components

### `ZImagePipeline`
- **Responsibility**: The high-level API for generation.
- **Capabilities**:
  - **Model Management**: Handles loading/unloading of individual components to manage memory usage.
  - **Memory Optimization**: Implements phase-scoped lifetimes. Text encoder is released immediately after embedding generation. Transformer is unloaded and cache cleared before VAE decoding to prevent memory spikes.
  - **AIO Support**: Detects and loads single-file "All-In-One" checkpoints (Transformer + Text Encoder + VAE) via `ZImageAIOCheckpoint`, bypassing base model weight loading.
  - **LoRA Support**: Dynamically loads/unloads LoRA adapters into the Transformer.
  - **Transformer Overrides**: Allows swapping the base transformer weights with a fine-tuned checkpoint (safetensors) without reloading the rest of the model.
  - **Prompt Enhancement**: Invokes the `QwenTextEncoder`'s generation capability to rewrite prompts before diffusion.
  - **CFG**: Implements standard Classifier-Free Guidance.
- **Flow**:
  1.  **Prep**: Resolve model paths, load components, compile LoRA.
  2.  **Enhance**: (Optional) Expand prompt via LLM.
  3.  **Encode**: Convert Text -> Embeddings (Positive & Negative).
  4.  **Init Latents**: Create random noise `(1, 16, h, w)`.
  5.  **Denoise**: Loop `N` steps using `FlowMatchEulerScheduler`.
  6.  **Decode**: Latents -> VAE -> Image.
  7.  **Save**: VAE Output -> PNG.

### `FlowMatchEulerScheduler`
- **Responsibility**: Controls the noise schedule and update step.
- **Algorithm**: Euler method for Flow Matching.
- **Feature**: **Dynamic Shifting**. Adjusts the time/noise schedule based on the target image resolution (`mu` parameter). This ensures consistent signal-to-noise ratios across different aspect ratios and resolutions.

### `ZImageGenerationRequest`
- **Responsibility**: Configuration object for a single generation job.
- **Fields**: Prompt, Negative Prompt, Dimensions, Steps, Guidance Scale, Seed, LoRA Config, etc.

## Data Flow
`User Prompt` -> `[TextEncoder]` -> `Embeddings`
`Random Noise` -> `[Transformer (Loop)]` <- `Embeddings`
                                      <- `Timestep`
`Refined Latents` -> `[VAE]` -> `Image`

## Dependencies
- Relies on all `Sources/ZImage/Model` components.
- Uses `HubApi` for model resolution.
- Uses `QwenImageIO` for saving results.

## Code Quality Observations (from Batch 1 & 6 Analysis)

### Sources/ZImage/Pipeline/ZImagePipeline.swift
- **Purpose**: Main text-to-image generation pipeline.
- **Key Responsibilities**:
  - Model loading/unloading (TextEncoder, Transformer, VAE, Tokenizer).
  - Memory management (explicit `unload` methods, `GPU.clearCache`).
  - Generation loop (Scheduler stepping).
  - Weight loading and canonicalization (handling AIO checkpoints, overrides).
- **Observations**:
  - **Complexity**: High. Handles file resolution, weight mapping, memory monitoring, and generation logic all in one place.
  - **Hardcoded Values**: Contains hardcoded model IDs (`areZImageVariants`).
  - **Duplication**: `canonicalizeTransformerOverride` logic is complex and specific.
  - **Performance**: Has specific optimizations for memory (unloading components when not needed).

### Sources/ZImage/Pipeline/ZImageControlPipeline.swift
- **Purpose**: Generation pipeline with ControlNet support.
- **Key Responsibilities**:
  - Similar to `ZImagePipeline` but adds `ControlNet` context (control images, masks).
  - Handles `inpaint` logic.
- **Observations**:
  - **High Duplication**: Significant overlap with `ZImagePipeline` (init, loading, memory management, scheduler loop).
  - **Internal Types**: Defines `ControlProgress` and `ZImageControlGenerationRequest` which are very similar to base pipeline equivalents.
  - **Weight Mapping**: Contains `ZImageControlWeightsMapping` enum with weight application logic. This seems misplaced; should be in `Weights` module.

### Sources/ZImage/Pipeline/PipelineSnapshot.swift
- **Purpose**: Helper to prepare model files (download/resolve).
- **Observations**:
  - Simple wrapper around `ModelResolution`.
  - Defines required file patterns.

### Sources/ZImage/Pipeline/FlowMatchScheduler.swift
- **Purpose**: Diffusion noise scheduler (Flow Matching Euler).
- **Observations**:
  - Clean implementation of the math.
  - Supports dynamic time shifting.

### Sources/ZImage/Pipeline/PipelineUtilities.swift
- **Purpose**: Shared helpers for pipelines.
- **Key Functions**:
  - `encodePrompt`: Wraps tokenization + encoding.
  - `decodeLatents`: Wraps VAE decode + image denormalization.
- **Observations**:
  - Reduces some duplication between standard and control pipelines.