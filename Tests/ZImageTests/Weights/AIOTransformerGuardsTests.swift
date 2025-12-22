import XCTest
import MLX
import Logging
@testable import ZImage

final class AIOTransformerGuardsTests: XCTestCase {

  private func makeConfig(qkNorm: Bool) -> ZImageTransformerConfig {
    ZImageTransformerConfig(
      inChannels: 4,
      dim: 4,
      nLayers: 1,
      nRefinerLayers: 0,
      nHeads: 1,
      nKVHeads: 1,
      normEps: 1e-5,
      qkNorm: qkNorm,
      capFeatDim: 4,
      ropeTheta: 10_000,
      tScale: 1.0,
      axesDims: [2],
      axesLens: [1]
    )
  }

  func testStrictAIORequiresQKNormSentinelsWhenEnabled() {
    let pipeline = ZImagePipeline(logger: Logger(label: "test"))
    let w = MLXArray([Float(0.0)])
    let weights: [String: MLXArray] = [
      "layers.0.attention.to_q.weight": w,
      "layers.0.attention.to_out.0.weight": w,
    ]

    XCTAssertThrowsError(try pipeline.validateStrictAIOTransformerWeights(weights, config: makeConfig(qkNorm: true))) { error in
      guard case ZImagePipeline.PipelineError.weightsMissing(let message) = error else {
        XCTFail("Unexpected error: \(error)")
        return
      }
      XCTAssertTrue(message.contains("norm_q.weight"))
      XCTAssertTrue(message.contains("norm_k.weight"))
    }
  }

  func testStrictAIOAllowsMissingQKNormWhenDisabled() throws {
    let pipeline = ZImagePipeline(logger: Logger(label: "test"))
    let w = MLXArray([Float(0.0)])
    let weights: [String: MLXArray] = [
      "layers.0.attention.to_q.weight": w,
      "layers.0.attention.to_out.0.weight": w,
    ]

    try pipeline.validateStrictAIOTransformerWeights(weights, config: makeConfig(qkNorm: false))
  }

  func testAIOTransformerCoverageThrowsWhenTooLow() {
    let pipeline = ZImagePipeline(logger: Logger(label: "test"))
    let config = makeConfig(qkNorm: true)
    let transformer = ZImageTransformer2DModel(configuration: config)
    let w = MLXArray([Float(0.0)])
    let weights: [String: MLXArray] = [
      "layers.0.attention.to_q.weight": w,
      "layers.0.attention.to_out.0.weight": w,
      "layers.0.attention.norm_q.weight": w,
      "layers.0.attention.norm_k.weight": w,
    ]

    XCTAssertThrowsError(try pipeline.validateAIOTransformerCoverage(weights, transformer: transformer, minimumCoverage: 0.99))
  }

  func testAIOTransformerCoveragePassesWhenFull() throws {
    let pipeline = ZImagePipeline(logger: Logger(label: "test"))
    let config = makeConfig(qkNorm: true)
    let transformer = ZImageTransformer2DModel(configuration: config)
    let placeholder = MLXArray([Float(0.0)])

    var weights: [String: MLXArray] = [:]
    for (key, _) in transformer.parameters().flattened() {
      weights[key] = placeholder
    }

    try pipeline.validateAIOTransformerCoverage(weights, transformer: transformer, minimumCoverage: 0.99)
  }

  func testAIOTransformerCoverageAcceptsCapEmbedderAliases() throws {
    let pipeline = ZImagePipeline(logger: Logger(label: "test"))
    let config = makeConfig(qkNorm: true)
    let transformer = ZImageTransformer2DModel(configuration: config)
    let placeholder = MLXArray([Float(0.0)])

    var weights: [String: MLXArray] = [:]
    for (key, _) in transformer.parameters().flattened() {
      weights[key] = placeholder
    }

    XCTAssertNotNil(weights.removeValue(forKey: "capEmbedNorm.weight"))
    XCTAssertNotNil(weights.removeValue(forKey: "capEmbedLinear.weight"))
    XCTAssertNotNil(weights.removeValue(forKey: "capEmbedLinear.bias"))
    weights["cap_embedder.0.weight"] = placeholder
    weights["cap_embedder.1.weight"] = placeholder
    weights["cap_embedder.1.bias"] = placeholder

    try pipeline.validateAIOTransformerCoverage(weights, transformer: transformer, minimumCoverage: 0.99)
  }
}
