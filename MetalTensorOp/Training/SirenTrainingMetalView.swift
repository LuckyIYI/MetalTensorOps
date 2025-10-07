import SwiftUI
import MetalKit

#if os(macOS)
struct SirenTrainingMetalView: NSViewRepresentable {
    @ObservedObject var coordinator: SirenTrainingCoordinator

    func makeCoordinator() -> SirenTrainingCoordinator { coordinator }

    func makeNSView(context: Context) -> MTKView {
        let view = MTKView()
        guard let device = MTLCreateSystemDefaultDevice() else {
            print("Failed to create Metal device")
            return view
        }
        configure(view: view, device: device, coordinator: context.coordinator)
        return view
    }

    func updateNSView(_ nsView: MTKView, context: Context) {
        if coordinator.groundTruthImage != nil {
            coordinator.updateDrawableSizes()
        }
    }
}
#else
struct SirenTrainingMetalView: UIViewRepresentable {
    @ObservedObject var coordinator: SirenTrainingCoordinator

    func makeCoordinator() -> SirenTrainingCoordinator { coordinator }

    func makeUIView(context: Context) -> MTKView {
        let view = MTKView()
        guard let device = MTLCreateSystemDefaultDevice() else {
            print("Failed to create Metal device")
            return view
        }
        configure(view: view, device: device, coordinator: context.coordinator)
        return view
    }

    func updateUIView(_ uiView: MTKView, context: Context) {
        if coordinator.groundTruthImage != nil {
            coordinator.updateDrawableSizes()
        }
    }
}
#endif

private func configure(view: MTKView, device: MTLDevice, coordinator: SirenTrainingCoordinator) {
    view.device = device
    view.framebufferOnly = false
    (view.layer as? CAMetalLayer)?.framebufferOnly = false
    view.clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1)
    view.delegate = coordinator
    coordinator.setup(device: device, mtkView: view)
}
