# Module: Quantization (`Sources/ZImage/Quantization`)

## Purpose
Provides tools to compress the model weights into lower precision (4-bit, 8-bit) to reduce memory footprint and storage requirements, enabling the model to run on consumer hardware with limited RAM.

## Key Components

### `ZImageQuantizer`
- **Responsibility**: The main utility for creating and loading quantized models.
- **Capabilities**:
  - **Quantize & Save**: Reads a full-precision model, quantizes eligible layers (Linear layers with suitable dimensions), and saves the new weights plus a manifest.
  - **Apply**: Reads a `ZImageQuantizationManifest` and transforms an in-memory `Module` (replacing `Linear` with `QuantizedLinear`) to match the data on disk.
  - **ControlNet Support**: Specialized logic for quantizing ControlNet adapter weights.

### `ZImageQuantizationManifest`
- **Responsibility**: A JSON schema (`quantization.json`) describing how the model was quantized.
- **Fields**: `groupSize` (32, 64, 128), `bits` (4, 8), `mode` (affine, mxfp4), and a list of layers.

### `ZImageQuantizationSpec`
- **Responsibility**: User configuration for the quantization process (target bits, group size).

## Techniques
- **Group-wise Quantization**: Weights are quantized in small groups (e.g., every 32 parameters share a scale/bias) to maintain accuracy.
- **Modes**:
  - **Affine**: Standard integer quantization with scale and zero-point.
  - **MXFP4**: Micro-exponent Floating Point 4-bit (newer standard, often higher quality for LLMs/Diffusion).

## Data Flow
- **Offline (CLI)**: `ZImageQuantizer.quantizeAndSave(...)` -> Produces `quantization.json` + `*.safetensors`.
- **Runtime**: `ZImageWeightsMapper` detects `quantization.json` -> calls `ZImageQuantizer.applyQuantization(...)` -> Model is ready for loading quantized weights.
