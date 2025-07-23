import SwiftUI
import Metal
import MetalKit

#if os(macOS)
import AppKit
#elseif os(iOS)
import UIKit
#endif

#if os(macOS)
struct MetalCircleView: NSViewRepresentable {
    func makeCoordinator() -> Coordinator { Coordinator() }
    func makeNSView(context: Context) -> MTKView {
        let mtkView = MTKView()
        guard let device = MTLCreateSystemDefaultDevice() else {
            print("[MetalCircleView] Failed to create system default Metal device")
            return mtkView
        }
        mtkView.device = device
        mtkView.framebufferOnly = false
        mtkView.delegate = context.coordinator
        mtkView.clearColor = .init(red: 1, green: 1, blue: 1, alpha: 1)
        context.coordinator.setup(device: device, mtkView: mtkView)

        return mtkView
    }
    func updateNSView(_ nsView: MTKView, context: Context) {}
}
#else
struct MetalCircleView: UIViewRepresentable {
    func makeCoordinator() -> Coordinator { Coordinator() }
    func makeUIView(context: Context) -> MTKView {
        let mtkView = MTKView()
        guard let device = MTLCreateSystemDefaultDevice() else {
            print("[MetalCircleView] Failed to create system default Metal device")
            return mtkView
        }
        mtkView.device = device
        mtkView.framebufferOnly = false
        mtkView.delegate = context.coordinator
        mtkView.clearColor = .init(red: 1, green: 1, blue: 1, alpha: 1)
        context.coordinator.setup(device: device, mtkView: mtkView)

        return mtkView
    }
    func updateUIView(_ uiView: MTKView, context: Context) {}
}
#endif

@available(macOS 26.0, iOS 16.0, *)
class Coordinator: NSObject, MTKViewDelegate {
    var device: MTLDevice?
    var commandQueue: MTL4CommandQueue?
    var commandAllocator: MTL4CommandAllocator?
    var encoder: MetalEncoder?
    var commandBuffer: MTL4CommandBuffer?
    var compiler: MTL4Compiler?

    private var drawingEnabled = false

    func setup(device: MTLDevice, mtkView: MTKView) {
        self.device = device

        guard let commandQueue = device.makeMTL4CommandQueue() else {
            print("[Coordinator] Failed to create MTL4CommandQueue")
            drawingEnabled = false
            return
        }
        self.commandQueue = commandQueue

        guard let commandAllocator = device.makeCommandAllocator() else {
            print("[Coordinator] Failed to create MTL4CommandAllocator")
            drawingEnabled = false
            return
        }
        self.commandAllocator = commandAllocator

        do {
            compiler = try device.makeCompiler(descriptor: .init())
        } catch {
            print("[Coordinator] Failed to create MTL4Compiler: \(error)")
            drawingEnabled = false
            return
        }

        let library: MTLLibrary
        do {
            library = try device.makeDefaultLibrary(bundle: .main)
        } catch {
            print("[Coordinator] Failed to create default library: \(error)")
            drawingEnabled = false
            return
        }

        guard let compiler = compiler else {
            print("[Coordinator] Compiler is nil after creation")
            drawingEnabled = false
            return
        }

        do {
            encoder = try MetalEncoder(device: device, library: library, compiler: compiler, queue: commandQueue)
        } catch {
            print("[Coordinator] Failed to initialize MetalEncoder: \(error)")
            drawingEnabled = false
            return
        }

        guard let commandBuffer = device.makeCommandBuffer() else {
            print("[Coordinator] Failed to create MTL4CommandBuffer")
            drawingEnabled = false
            return
        }
        self.commandBuffer = commandBuffer

        guard let metalLayer = mtkView.layer as? CAMetalLayer else {
            print("[Coordinator] MTKView's layer is not a CAMetalLayer")
            drawingEnabled = false
            return
        }

        commandQueue.addResidencySet(metalLayer.residencySet)

        mtkView.drawableSize = CGSize(width: 512, height: 512)

        drawingEnabled = true
    }

    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {}

    func draw(in view: MTKView) {
        guard drawingEnabled else {
            return
        }

        guard let drawable = view.currentDrawable else {
            print("[Coordinator] No current drawable available")
            return
        }
        guard let commandAllocator = commandAllocator else {
            print("[Coordinator] Command allocator is nil")
            return
        }
        guard let commandBuffer = device?.makeCommandBuffer() else {
            print("[Coordinator] Failed to create MTL4CommandBuffer in draw")
            return
        }
        guard let encoder = encoder else {
            print("[Coordinator] Encoder is nil")
            return
        }
        guard let commandQueue = commandQueue else {
            print("[Coordinator] Command queue is nil")
            return
        }

        commandAllocator.reset()
        commandBuffer.beginCommandBuffer(allocator: commandAllocator)

        encoder.encode(drawableTexture: drawable.texture, commandBuffer: commandBuffer)
        commandBuffer.endCommandBuffer()

        commandQueue.waitForDrawable(drawable)
        commandQueue.commit([commandBuffer])
        commandQueue.signalDrawable(drawable)
        drawable.present()
    }
}

@available(macOS 26.0, iOS 16.0, *)
struct ContentView: View {
    var body: some View {
        MetalCircleView()
            .frame(width: 512, height: 512)
            .padding()
    }
}

#Preview {
    ContentView()
}
