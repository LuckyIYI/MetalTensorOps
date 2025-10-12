import SwiftUI

private let isRunningTests: Bool = {
    ProcessInfo.processInfo.environment["XCTestConfigurationFilePath"] != nil
}()

@main
struct MetalTensorOpApp: App {
    @State private var selectedModel: ModelKind = .instantNGP
    @State private var selectedRenderMode: RenderMode = .cooperative

    var body: some Scene {
        WindowGroup {
            if !isRunningTests {
                NavigationView {
                    VStack {
                        Picker("Neural Network", selection: $selectedModel) {
                            ForEach(ModelKind.allCases, id: \.self) { model in
                                Text(model.rawValue).tag(model)
                            }
                        }
                        .pickerStyle(.segmented)
                        .padding()

                        Picker("Render Mode", selection: $selectedRenderMode) {
                            ForEach(RenderMode.allCases) { mode in
                                Text(mode.rawValue).tag(mode)
                            }
                        }
                        .pickerStyle(.segmented)
                        .padding([.horizontal, .bottom])

                        ContentView()
                            .environment(\.modelKind, selectedModel)
                            .environment(\.renderMode, selectedRenderMode)

                        Spacer()

                    }
                    .navigationTitle("MetalTensorOp")
                }
            } else {
                Text("Testing")
            }
        }
    }
}
