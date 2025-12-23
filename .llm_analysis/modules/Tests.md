# Module: Tests (`Tests/`)

## Purpose
Ensures regression safety, performance tracking, and correctness of core components.

## Code Quality Observations

### CLIEndToEndTests.swift
- **Purpose**: End-to-end tests for ZImageCLI command line interface.
- **Observations**:
  - Runs the built binary against test inputs.
  - Good regression safety for CLI arguments and overall flow.

### PerformanceTests.swift
- **Purpose**: Performance tests for Z-Image pipeline.
- **Observations**:
  - Tracks inference speed (time) and memory usage.
  - Critical for this performance-oriented project (Apple Silicon).

### SafeTensorsReaderTests.swift
- **Purpose**: Unit tests for the custom SafeTensors parser.
- **Observations**:
  - Verifies the custom parser against known/generated files.
  - Ensures robust handling of the weights format.
