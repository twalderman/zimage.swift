# Design Patterns

## Repository / Gateway
`ModelResolution` and `HubSnapshot` act as a gateway to external model repositories (Hugging Face), abstracting away the details of network requests and caching.

## Adapter / Mapper
`ZImageWeightsMapper` and `ZImageWeightsMapping` function as adapters, converting the external schema (PyTorch/SafeTensors naming) into the internal domain schema (MLX Module hierarchy).

## Strategy / Reader
`SafeTensorsReader` implements a specific strategy for reading weight files, isolating the binary format parsing from the logic of how those weights are used.

## Dual-Mode Model
The `QwenTextEncoder` exhibits a dual-mode pattern:
1.  **Representation**: Acts as a pure encoder (returning hidden states) for the diffusion process.
2.  **Generation**: Acts as a causal language model (returning next-token logits) for the prompt enhancement process.