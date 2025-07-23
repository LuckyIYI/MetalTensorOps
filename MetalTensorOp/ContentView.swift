import SwiftUI
import Combine
import Metal
import MetalKit

struct ModelKindKey: EnvironmentKey {
    static let defaultValue: ModelKind = .siren
}

extension EnvironmentValues {
    var modelKind: ModelKind {
        get { self[ModelKindKey.self] }
        set { self[ModelKindKey.self] = newValue }
    }
}

protocol ComputeEncoder {
    func encode(drawableTexture: MTLTexture, commandBuffer: MTL4CommandBuffer)
}

#if os(macOS)
import AppKit
#elseif os(iOS)
import UIKit
#endif

#if os(macOS)
struct MetalCircleView: NSViewRepresentable {
    var coordinator: Coordinator

    init(coordinator: Coordinator) {
        self.coordinator = coordinator
    }

    func makeCoordinator() -> Coordinator { coordinator }

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

        if let width = context.coordinator.imageWidth, let height = context.coordinator.imageHeight {
            mtkView.drawableSize = CGSize(width: width, height: height)
        } else {
            mtkView.drawableSize = CGSize(width: 512, height: 512)
        }

        return mtkView
    }

    func updateNSView(_ nsView: MTKView, context: Context) {}
}
#else
struct MetalCircleView: UIViewRepresentable {
    var coordinator: Coordinator

    init(coordinator: Coordinator) {
        self.coordinator = coordinator
    }

    func makeCoordinator() -> Coordinator { coordinator }

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

        if let width = context.coordinator.imageWidth, let height = context.coordinator.imageHeight {
            mtkView.drawableSize = CGSize(width: width, height: height)
        } else {
            mtkView.drawableSize = CGSize(width: 512, height: 512)
        }

        return mtkView
    }

    func updateUIView(_ uiView: MTKView, context: Context) {}
}
#endif

@available(macOS 26.0, iOS 16.0, *)
class Coordinator: NSObject, MTKViewDelegate, ObservableObject {
    var device: MTLDevice?
    var commandQueue: MTL4CommandQueue?
    var commandAllocator: MTL4CommandAllocator?
    var encoder: ComputeEncoder?
    var commandBuffer: MTL4CommandBuffer?
    var compiler: MTL4Compiler?

    @Published var imageWidth: Int?
    @Published var imageHeight: Int?
    @Published var aspectRatio: CGFloat?

    private var drawingEnabled = false

    let modelKind: ModelKind

    init(modelKind: ModelKind = .siren) {
        self.modelKind = modelKind
        super.init()
    }

    struct ModelFile: Decodable {
        struct Metadata: Decodable {
            struct ImageMetadata: Decodable {
                let width: Int?
                let height: Int?
            }
            let image: ImageMetadata?
        }
        let metadata: Metadata?
    }

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
            switch modelKind {
            case .siren:
                encoder = try SirenEncoder(device: device, library: library, compiler: compiler, queue: commandQueue)
            case .fourier:
                encoder = try FourierEncoder(device: device, library: library, compiler: compiler, queue: commandQueue)
            }
        } catch {
            print("[Coordinator] Failed to initialize encoder for \(modelKind): \(error)")
            drawingEnabled = false
            return
        }

        if let url = Bundle.main.url(forResource: modelKind == .fourier ? "fourier" : "siren", withExtension: "json") {
            if let data = try? Data(contentsOf: url),
               let model = try? JSONDecoder().decode(ModelFile.self, from: data),
               let imageMeta = model.metadata?.image {
                if let width = imageMeta.width, let height = imageMeta.height {
                    self.imageWidth = width
                    self.imageHeight = height
                    self.aspectRatio = CGFloat(width) / CGFloat(height)
                }
            }
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

        if let width = imageWidth, let height = imageHeight {
            mtkView.drawableSize = CGSize(width: width, height: height)
        } else {
            mtkView.drawableSize = CGSize(width: 512, height: 512)
        }

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
    @Environment(\.modelKind) private var modelKind

    var body: some View {
        let coordinator = Coordinator(modelKind: modelKind)
        let width = coordinator.imageWidth.map { CGFloat($0) } ?? 512
        let height = coordinator.imageHeight.map { CGFloat($0) } ?? 512

        MetalCircleView(coordinator: coordinator)
            .frame(width: width, height: height)
            .padding()
            .id(modelKind)
    }
}

#Preview {
    ContentView()
        .environment(\.modelKind, .siren) // or .fourier for preview purposes
}
