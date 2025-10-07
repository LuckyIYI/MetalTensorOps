import Foundation
import Metal

protocol ComputeEncoder {
    func encode(drawableTexture: MTLTexture, commandBuffer: MTL4CommandBuffer, mode: RenderMode)
    func supports(_ mode: RenderMode) -> Bool
}

extension ComputeEncoder {
    func supports(_ mode: RenderMode) -> Bool { mode == .perPixel }
}

#if !TRAINING_CLI
enum ModelKind: String, CaseIterable, Identifiable {
    case siren = "Siren"
    case fourier = "Fourier"
    case instantNGP = "Instant NGP"

    var id: String { rawValue }
}
#endif
