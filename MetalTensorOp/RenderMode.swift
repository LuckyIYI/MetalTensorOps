import Foundation

#if canImport(SwiftUI) && !TRAINING_CLI
import SwiftUI
#endif

enum RenderMode: String, CaseIterable, Identifiable, Hashable {
    case perPixel = "Per Pixel"
    case cooperative = "Cooperative"

    var id: String { rawValue }
}
#if canImport(SwiftUI) && !TRAINING_CLI
struct RenderModeKey: EnvironmentKey {
    static let defaultValue: RenderMode = .perPixel
}

extension EnvironmentValues {
    var renderMode: RenderMode {
        get { self[RenderModeKey.self] }
        set { self[RenderModeKey.self] = newValue }
    }
}
#endif
