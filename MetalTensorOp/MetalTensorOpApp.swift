import SwiftUI

private let isRunningTests: Bool = {
    ProcessInfo.processInfo.environment["XCTestConfigurationFilePath"] != nil
}()

@main
struct MetalTensorOpApp: App {
    var body: some Scene {
        WindowGroup {
            if isRunningTests {
                Text("Testing")
            } else {
                ContentView()
            }
        }
    }
}
