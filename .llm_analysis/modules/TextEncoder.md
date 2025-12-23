# Module: Text Encoder (`Sources/ZImage/Model/TextEncoder`)

## Purpose
This module implements the Qwen-based text encoder used to condition the diffusion generation. Uniquely, it also includes generation capabilities to act as a "Prompt Enhancer," rewriting user prompts before they reach the diffusion stage.

## Key Components

### `QwenTextEncoder`
- **Responsibility**: The main wrapper class.
- **Capabilities**:
  - **Encoding**: Converts input tokens into embeddings.
  - **Z-Image Specific Encoding**: `encodeForZImage` extracts the **second-to-last hidden state**, a common technique in advanced diffusion models (like SDXL, Flux) to capture richer semantic information than the final layer.
  - **Joint Encoding**: `encodeJoint` supports multimodal conditioning by splicing vision tokens (from a vision tower) into the text embedding sequence.
  - **Prompt Enhancement**: `enhancePrompt` runs the model in autoregressive mode (LLM style) to rewrite short user prompts into detailed descriptive prompts using a system prompt.

### `QwenEncoder` & `QwenEncoderLayer`
- **Architecture**: Standard Transformer Decoder architecture (despite the name "Encoder", it uses Causal Masking for generation and RoPE).
- **Components**:
  - `QwenAttention`: Self-attention with RoPE (Rotary Positional Embeddings) and RMSNorm.
  - `QwenMLP`: SwiGLU-style MLP.
  - `RMSNorm`: Pre-normalization.

### `QwenGeneration` (Extension)
- **Responsibility**: Autoregressive text generation.
- **Features**:
  - `KVCache`: Simple key-value cache implementation for efficient step-by-step generation.
  - `generate()`: Greedy/Sampling loop.
  - **System Prompt**: Contains a hardcoded "Prompt Enhancer" system prompt that instructs the model to act as a "visionary artist" to expand prompts.

## Usage Pattern
1. **Prompt Enhancement (Optional)**: User input -> `enhancePrompt` -> Expanded Text.
2. **Conditioning**: Expanded Text -> `encodeForZImage` -> Embeddings (Second-to-last layer).
3. **Multimodal (Optional)**: Image + Text -> `encodeJoint` -> Hybrid Embeddings.

## Code Quality Observations

### Sources/ZImage/Model/TextEncoder/TextEncoder.swift
- **Purpose**: Qwen-based Text Encoder.
- **Roles**:
  1.  **Embedding Provider**: Extracts hidden states (2nd to last layer) for DiT conditioning.
  2.  **Prompt Enhancer**: Can generate text autoregressively to refine user prompts.
- **Key Methods**:
  - `encodeForZImage`: Specialized extraction logic.
  - `enhancePrompt`: Uses LLM generation to rewrite prompts.
  - `encodeJoint`: Multimodal support (replacing placeholder tokens with image embeddings).

### Sources/ZImage/Model/TextEncoder/LLMGeneration/QwenGeneration.swift
- **Purpose**: Autoregressive generation logic for Qwen.
- **Observations**:
  - **Hardcoded System Prompt**: `peSystemPrompt` defines the persona for prompt enhancement ("Logic Prison Vision Artist").
  - **Generation Loop**: Standard `generate` function with sampling.