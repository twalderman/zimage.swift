# Module: ZImageCLI (`Sources/ZImageCLI`)

## Purpose
The command-line interface entry point for the project. It exposes the library's capabilities (Generation, ControlNet, Quantization) to the user via terminal commands.

## Key Components

### `ZImageCLI`
- **Responsibility**: Main executable logic.
- **Argument Parsing**: Custom manual parser (iterates `CommandLine.arguments`).
- **Commands**:
  - **Generation**: `ZImageCLI -p "..."`. Uses `ZImagePipeline`.
  - **ControlNet**: `ZImageCLI control ...`. Uses `ZImageControlPipeline`.
  - **Quantization**: `ZImageCLI quantize ...` / `quantize-controlnet`. Uses `ZImageQuantizer`.
- **Progress Reporting**: Custom `ProgressBar` (TTY-aware) and `PlainProgress` (for logs/CI).

## Usage Pattern
The CLI initializes the appropriate Pipeline or Quantizer based on arguments, sets up logging/progress handlers, runs the task, and handles errors. It also manages global MLX settings like `GPU.set(cacheLimit:)`.
