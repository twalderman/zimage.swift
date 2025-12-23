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

## Code Quality Observations
### Sources/ZImageCLI/main.swift
- **Key Functions**:
  - `run()`: Main execution loop, parses args, sets up logging/GPU, triggers pipeline.
  - `runQuantize()`, `runQuantizeControlnet()`: Handles quantization commands.
  - `runControl()`: Handles ControlNet generation.
  - `printUsage()`: Help text.
- **Observations**:
  - **Manual Argument Parsing**: Uses `Iterator` over `CommandLine.arguments`. This is fragile and reinventing the wheel compared to `swift-argument-parser`.
  - **Logic**: Handles device selection (Metal vs CPU) and basic configuration parsing.
  - **Duplication**: Similar parsing logic for different commands (generate vs control vs quantize).

### Package.swift
- **Purpose**: Defines the Swift package structure, dependencies, and targets.
- **Dependencies**: 
  - `mlx-swift` (Machine Learning)
  - `swift-transformers` (Text tokenization/encoding)
  - `swift-log` (Logging)
- **Targets**:
  - `ZImage`: Core library.
  - `ZImageCLI`: Command-line executable.
  - Tests: Unit (`ZImageTests`), Integration (`ZImageIntegrationTests`), E2E (`ZImageE2ETests`).