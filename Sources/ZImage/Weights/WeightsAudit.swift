import Foundation
import Logging
import MLXNN
import MLX

struct WeightsAudit {
  struct Summary {
    let matched: Int
    let missing: [String]
    let extra: [String]
  }

  static func audit(module: Module, weights: [String: MLXArray], prefix: String = "", logger: Logger, sample: Int = 5) -> Summary {
    let params = module.parameters().flattened()
    var matched = 0
    var missingKeys: [String] = []
    var remaining = Set(weights.keys)

    for (key, _) in params {
      let candidate1 = key
      let candidate2 = prefix.isEmpty ? key : "\(prefix).\(key)"
      if weights[candidate2] != nil {
        matched += 1
        remaining.remove(candidate2)
      } else if weights[candidate1] != nil {
        matched += 1
        remaining.remove(candidate1)
      } else {
        missingKeys.append(candidate2)
      }
    }

    let missingSample = Array(missingKeys.prefix(max(0, sample)))
    let extraSample = Array(Array(remaining).sorted().prefix(max(0, sample)))

    logger.info("\(prefix.isEmpty ? "module" : prefix) weights audit -> matched: \(matched), missing: \(missingKeys.count), extra: \(remaining.count)")
    if !missingKeys.isEmpty {
      let suffix = missingKeys.count > missingSample.count ? ", ..." : ""
      let sampleText = missingSample.isEmpty ? "" : " (sample: \(missingSample.joined(separator: ", "))\(suffix))"
      logger.warning("Missing weights: \(missingKeys.count)\(sampleText)")
    }
    if !remaining.isEmpty {
      let suffix = remaining.count > extraSample.count ? ", ..." : ""
      let sampleText = extraSample.isEmpty ? "" : " (sample: \(extraSample.joined(separator: ", "))\(suffix))"
      logger.info("Extra weights: \(remaining.count)\(sampleText)")
    }

    return Summary(matched: matched, missing: missingKeys, extra: Array(remaining))
  }
}
