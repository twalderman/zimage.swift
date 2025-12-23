# Module: Tokenizer (`Sources/ZImage/Tokenizer`)

## Purpose
Wraps the Hugging Face `Tokenizers` library to provide text tokenization for the Qwen-based Text Encoder.

## Key Components

### `QwenTokenizer`
- **Responsibility**: Loads the tokenizer and converts strings to integer token IDs.
- **Backend**: Uses `AutoTokenizer` from `swift-transformers`.
- **Modes**:
  1.  **Chat Encoding**: `encodeChat` applies the Qwen chat template (`<|im_start|>user...`). Used for "Prompt Enhancement" and potentially instruction-tuning tasks.
  2.  **Plain Encoding**: `encodePlain` converts text directly without templates. Used for the diffusion conditioning embedding.
- **Templates**: Contains hardcoded prefixes/suffixes (likely for an image-editing instruction mode), though primarily used via the plain encoding path for standard generation.

## Data Flow
- `String` -> `QwenTokenizer` -> `QwenTokenBatch` (`inputIds`, `attentionMask`) -> `QwenTextEncoder`

## Dependencies
- `Sources/ZImage/Tokenizer` -> `Tokenizers` (swift-transformers)
- `Sources/ZImage/Tokenizer` -> `MLX`

## Code Quality Observations

### Sources/ZImage/Tokenizer/Tokenizer.swift
- **Purpose**: Wrapper around `swift-transformers` Tokenizer for Qwen.
- **Observations**:
  - **Special Tokens**: Handles Qwen specific tokens (`<|im_start|>`, etc.).
  - **Dual Mode**: `encodeChat` (for enhancement) vs `encodePlain` (for conditioning).