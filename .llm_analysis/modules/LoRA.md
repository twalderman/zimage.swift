# Module: LoRA (`Sources/ZImage/LoRA`)

## Purpose
Provides support for Low-Rank Adaptation (LoRA) and Low-Rank Kronecker Product (LoKr) fine-tuning adapters. This allows the model to be stylized or specialized without retraining the massive base model.

## Key Components

### `LoRAApplicator`
- **Responsibility**: The engine for applying LoRA weights.
- **Modes**:
  1.  **Merged (Static)**: Modifies the underlying weight tensors directly (`weight = weight + scale * up @ down`).
      - *Pros*: Zero inference overhead.
      - *Cons*: Slow switching (requires reloading or math on all weights), tricky with quantization (requires dequant -> merge -> requant).
  2.  **Dynamic (Runtime)**: Replaces standard layers with LoRA-aware wrappers.
      - *Pros*: Instant switching, memory efficient (keeps base weights frozen/quantized).
      - *Cons*: Slight inference overhead (extra matmuls).
- **LoKr Support**: Implements Kronecker product weight application (`w1 (kron) w2`).

### `LoRALinear` / `LoRAQuantizedLinear`
- **Responsibility**: Drop-in replacements for `Linear` and `QuantizedLinear` that implement the Dynamic LoRA logic.
- **Logic**: `Output = BaseLayer(x) + Scale * (x @ Down.T @ Up.T)`.

### `LoRAConfiguration`
- **Responsibility**: Defines the source (Local/HF) and scale of the LoRA to be applied.

## Data Flow
- **Loading**: Pipeline requests LoRA load.
- **Parsing**: `LoRAWeightLoader` (external to this module, likely in `Weights`) loads tensors.
- **Application**: `LoRAApplicator` walks the model hierarchy.
  - Matches LoRA keys to Model keys.
  - Swaps `Linear` -> `LoRALinear`.
  - Injects `down`, `up`, and `scale` tensors.

## Unique Characteristics
- **Quantization Aware**: The dynamic implementation allows applying high-precision (Float16/32) LoRA adapters on top of low-precision (Int4/8) quantized base weights without dequantizing the base model. This is critical for running on consumer hardware (Macs with limited RAM).

## Code Quality Observations

### Sources/ZImage/LoRA/LoRAApplicator.swift
- **Purpose**: Applies LoRA (Low-Rank Adaptation) weights to the model.
- **Capabilities**:
  - **Static Merging**: Fuses weights permanently.
  - **Dynamic Application**: Swaps layers for `LoRALinear`/`LoRAQuantizedLinear` at runtime.
  - **LoKr Support**: Handles LyCORIS/LoKr format (Kronecker product).
  - **Quantization Support**: Can apply LoRA on top of quantized base weights (dequantize -> merge -> requantize).

### Sources/ZImage/LoRA/LoRAWeightLoader.swift
- **Purpose**: Loads LoRA weights.
- **Observations**:
  - Supports local paths and HuggingFace IDs.
  - Heuristic detection of Down/Up pairs.
  - Parsing logic for LoKr weights.

### Sources/ZImage/LoRA/LoRAKeyMapper.swift
- **Purpose**: Translates external LoRA key formats (diffusers, kohya) to internal model keys.
- **Observations**:
  - Large hardcoded dictionaries.
  - Regex-like string replacement logic.
  - Critical for compatibility but high maintenance.