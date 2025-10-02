import SwiftUI

private let isRunningTests: Bool = {
    ProcessInfo.processInfo.environment["XCTestConfigurationFilePath"] != nil
}()

@main
struct MetalTensorOpApp: App {
    @State private var selectedModel: ModelKind = .instantNGP

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

                        ContentView()
                            .environment(\.modelKind, selectedModel)

                        Spacer()

                        VStack(alignment: .leading, spacing: 8) {
                            Text("Neural Rendering Comparison")
                                .font(.headline)
                            Text("• Siren: Sinusoidal activation networks")
                                .font(.caption)
                            Text("• Fourier: Positional encoding + ReLU")
                                .font(.caption)
                            Text("• Instant NGP: Hash encoding + cooperative MLP")
                                .font(.caption)
                                .foregroundColor(.green)
                        }
                        .padding()
                        .background(Color.gray.opacity(0.1))
                        .cornerRadius(10)
                        .padding()
                    }
                    .navigationTitle("MetalTensorOp")
                }
            } else {
                Text("Testing")
            }
        }
    }
}
