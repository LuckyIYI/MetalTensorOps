#if !TRAINING_CLI
import SwiftUI
import MetalKit

#if os(macOS)
struct RenderMetalView: NSViewRepresentable {
    @ObservedObject var coordinator: RenderCoordinator

    func makeCoordinator() -> RenderCoordinator { coordinator }

    func makeNSView(context: Context) -> MTKView {
        let view = MTKView()
        guard let device = MTLCreateSystemDefaultDevice() else {
            print("RenderMetalView: failed to create Metal device")
            return view
        }
        configure(view: view, device: device, coordinator: context.coordinator)
        return view
    }

    func updateNSView(_ nsView: MTKView, context: Context) {}
}
#else
struct RenderMetalView: UIViewRepresentable {
    @ObservedObject var coordinator: RenderCoordinator

    func makeCoordinator() -> RenderCoordinator { coordinator }

    func makeUIView(context: Context) -> MTKView {
        let view = MTKView()
        guard let device = MTLCreateSystemDefaultDevice() else {
            print("RenderMetalView: failed to create Metal device")
            return view
        }
        configure(view: view, device: device, coordinator: context.coordinator)
        return view
    }

    func updateUIView(_ uiView: MTKView, context: Context) {}
}
#endif

private func configure(view: MTKView, device: MTLDevice, coordinator: RenderCoordinator) {
    view.device = device
    view.framebufferOnly = false
    (view.layer as? CAMetalLayer)?.framebufferOnly = false
    view.isPaused = false
    view.enableSetNeedsDisplay = false
    view.clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1)
    view.delegate = coordinator
    coordinator.setup(device: device, mtkView: view)
}
#endif
