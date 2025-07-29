import SwiftUI

enum ModelKind {
    case siren
    case fourier
}

@main
struct MetalTensorOpApp: App {
    
    var body: some Scene {
        WindowGroup {
            ContentView().environment(\.modelKind, .fourier)
        }
    }
}
