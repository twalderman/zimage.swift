# Module: Support & Utils (`Sources/ZImage/Support`, `Sources/ZImage/Util`)

## Purpose
Provides essential utilities for image processing and defines static metadata for the supported model architecture.

## Key Components

### `ZImageModelMetadata`
- **Responsibility**: Hosting static configuration constants for the `Tongyi-MAI/Z-Image-Turbo` model.
- **Data**:
  - Model dimensions (hidden sizes, layer counts, head counts) for Transformer, TextEncoder, and VAE.
  - Recommended inference parameters (width, height, steps, guidance scale).
  - License and repository info.
- **Role**: Acts as a source of truth for defaults, likely used when dynamic config loading is incomplete or for validation.

### `QwenImageIO`
- **Responsibility**: Handles input/output operations for images, bridging the gap between Apple's `CoreGraphics` and `MLXArray`.
- **Key Features**:
  - **Conversion**: `CGImage` <-> `MLXArray` (handling channel ordering and normalization).
  - **Normalization**: Maps `[0, 1]` images to `[-1, 1]` for the model encoder and back.
  - **Resizing**: Implements high-quality Lanczos resampling (both via CoreGraphics and a custom pure-Swift implementation for `MLXArray` manipulation).
  - **Dependencies**: `CoreGraphics`, `ImageIO`, `UniformTypeIdentifiers`.

## Usage Pattern
- Pipelines use `QwenImageIO` to preprocess user images before feeding them to the VAE.
- Pipelines use `QwenImageIO` to save the generated `MLXArray` output as PNG files.
- `ZImageModelMetadata` is likely accessed during pipeline initialization to set default parameters.

## Code Quality Observations (from Batch 2 & 6 Analysis)

### Sources/ZImage/Support/ModelMetadata.swift
- **Purpose**: Static constants defining the Z-Image architecture.
- **Observations**:
  - Hardcoded dimensions (hidden sizes, layer counts, recommended resolutions).
  - Limits library generality to this specific model architecture.

### Sources/ZImage/Util/ImageIO.swift
- **Purpose**: Image loading, saving, and resizing.
- **Dependencies**: `CoreGraphics`, `ImageIO`.
- **Observations**:
  - **Manual Algorithms**: Contains a full Swift implementation of Lanczos resampling (`resizeLanczos`, `makeContributions`). This is likely to match PIL/PyTorch behavior exactly but reinvents OS capabilities.
  - **Platform Lock**: Heavily tied to Apple frameworks (`CGImage`, `CGContext`).