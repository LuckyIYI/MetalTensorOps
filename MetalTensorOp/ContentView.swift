import SwiftUI
import Combine
import Metal
import MetalKit

enum ModelKind: String, CaseIterable {
    case siren = "Siren"
    case fourier = "Fourier"
    case instantNGP = "Instant NGP"
}

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
        (mtkView.layer as? CAMetalLayer)?.framebufferOnly = false
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
        (mtkView.layer as? CAMetalLayer)?.framebufferOnly = false
        mtkView.delegate = context.coordinator
        mtkView.clearColor = .init(red: 0, green: 0, blue: 0, alpha: 1)
        mtkView.backgroundColor = .black
    
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
    var commandAllocators: [MTL4CommandAllocator] = []
    
    var sharedEvent: MTLSharedEvent?
    var frameNumber: UInt64 = 0
    let maxFramesInFlight = 3
    
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

        commandAllocators = []
        for _ in 0..<maxFramesInFlight {
            guard let allocator = device.makeCommandAllocator() else {
                print("[Coordinator] Failed to create MTL4CommandAllocator")
                drawingEnabled = false
                return
            }
            commandAllocators.append(allocator)
        }

        sharedEvent = device.makeSharedEvent()
        sharedEvent?.signaledValue = 0

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

        imageWidth = nil
        imageHeight = nil
        aspectRatio = nil

        do {
            switch modelKind {
            case .siren:
                encoder = try SirenEncoder(device: device, library: library, compiler: compiler, queue: commandQueue)
                loadMetadata(resourceName: "siren")
            case .fourier:
                encoder = try FourierEncoder(device: device, library: library, compiler: compiler, queue: commandQueue)
                loadMetadata(resourceName: "fourier")
            case .instantNGP:
                guard let weightsFile = loadInstantNGPWeightsFile() else {
                    print("[Coordinator] Instant NGP weights unavailable")
                    drawingEnabled = false
                    return
                }

                let metalWeights = try weightsFile.makeMetalWeights(device: device)
                let instantEncoder = try InstantNGPEncoder(
                    device: device,
                    library: library,
                    compiler: compiler,
                    queue: commandQueue,
                    weights: metalWeights
                )
                if let image = weightsFile.metadata.image,
                   let width = image.width,
                   let height = image.height {
                    imageWidth = width
                    imageHeight = height
                    aspectRatio = CGFloat(width) / CGFloat(height)
                }
                encoder = instantEncoder
            }
        } catch {

            print("[Coordinator] Failed to initialize encoder for \(modelKind): \(error)")
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

        if let width = imageWidth, let height = imageHeight {
            mtkView.drawableSize = CGSize(width: width, height: height)
        } else {
            mtkView.drawableSize = CGSize(width: 300, height: 300)
        }

        drawingEnabled = true
    }

    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {}

    func draw(in view: MTKView) {
        guard drawingEnabled else {
            return
        }
        
        if frameNumber >= maxFramesInFlight {
            let previousValueToWaitFor = frameNumber - UInt64(maxFramesInFlight)
            sharedEvent?.wait(untilSignaledValue: previousValueToWaitFor, timeoutMS: 10)
        }

        guard let drawable = view.currentDrawable else {
            print("[Coordinator] No current drawable available")
            return
        }
        
        let frameIndex = Int(frameNumber % UInt64(maxFramesInFlight))
        let commandAllocator = commandAllocators[frameIndex]
        commandAllocator.reset()

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

        commandBuffer.beginCommandBuffer(allocator: commandAllocator)

        encoder.encode(drawableTexture: drawable.texture, commandBuffer: commandBuffer)
        commandBuffer.endCommandBuffer()

        commandQueue.waitForDrawable(drawable)
        commandQueue.commit([commandBuffer])
        commandQueue.signalDrawable(drawable)
        drawable.present()
        
        let valueToSignal = frameNumber
        commandQueue.signalEvent(sharedEvent!, value: valueToSignal)
        
        frameNumber += 1
    }
}

extension Coordinator {
    private func loadMetadata(resourceName: String) {
        guard let url = Bundle.main.url(forResource: resourceName, withExtension: "json") else {
            return
        }

        guard let data = try? Data(contentsOf: url),
              let model = try? JSONDecoder().decode(ModelFile.self, from: data),
              let imageMeta = model.metadata?.image,
              let width = imageMeta.width,
              let height = imageMeta.height else {
            return
        }

        imageWidth = width
        imageHeight = height
        aspectRatio = CGFloat(width) / CGFloat(height)
    }

    private func loadInstantNGPWeightsFile() -> InstantNGPWeightsFile? {
        guard let url = Bundle.main.url(forResource: "instant_ngp", withExtension: "json") else {
            print("[Coordinator] instant_ngp.json not found in bundle")
            return nil
        }

        do {
            let weights = try InstantNGPWeightsFile.load(from: url)
            return weights
        } catch {
            print("[Coordinator] Failed to load Instant NGP weights: \(error)")
            return nil
        }
    }
}

@available(macOS 26.0, iOS 16.0, *)
struct ContentView: View {
    @Environment(\.modelKind) private var modelKind

    var body: some View {
        let coordinator = Coordinator(modelKind: modelKind)
        let width = coordinator.imageWidth.map { CGFloat($0) } ?? 300
        let height = coordinator.imageHeight.map { CGFloat($0) } ?? 300
        
        ZStack {
            Color.black.ignoresSafeArea()
            MetalCircleView(coordinator: coordinator)
                .frame(width: width, height: height)
                .id(modelKind)
        }
    }
}

#Preview {
    ContentView()
        .environment(\.modelKind, .siren)
}
