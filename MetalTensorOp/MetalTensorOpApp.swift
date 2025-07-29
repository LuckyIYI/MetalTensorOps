import SwiftUI

private let isRunningTests: Bool = {
    ProcessInfo.processInfo.environment["XCTestConfigurationFilePath"] != nil
}()

enum ModelKind {
    case siren
    case fourier
}

@main
struct MetalTensorOpApp: App {
    
    var body: some Scene {
        WindowGroup {
            if !isRunningTests {
                ContentView().environment(\.modelKind, .fourier)
            } else {
                Text("Testing")
            }
        }
    }
}
