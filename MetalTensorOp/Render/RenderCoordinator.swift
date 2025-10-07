#if !TRAINING_CLI
import Foundation
import CoreGraphics
import Metal
import MetalKit
import Combine

final class RenderCoordinator: NSObject, MTKViewDelegate, ObservableObject {
    @Published private(set) var imageSize: CGSize = .zero
    @Published private(set) var aspectRatio: CGFloat = 1
    @Published private(set) var isReady: Bool = false

    private var modelKind: ModelKind = .siren
    private var renderMode: RenderMode = .perPixel

    private var device: MTLDevice?
    private var commandQueue: MTL4CommandQueue?
    private var commandAllocators: [MTL4CommandAllocator] = []
    private var sharedEvent: MTLSharedEvent?
    private var compiler: MTL4Compiler?
    private weak var metalView: MTKView?
    private var encoder: ComputeEncoder?

    private var frameNumber: UInt64 = 0
    private let maxFramesInFlight = 3
    private var drawingEnabled = false

    func setup(device: MTLDevice, mtkView: MTKView) {
        self.device = device
        self.metalView = mtkView

        do {
            guard let queue = device.makeMTL4CommandQueue() else {
                throw RendererError.makeQueueFailed
            }
            commandQueue = queue

            commandAllocators = try (0..<maxFramesInFlight).map { _ in
                guard let allocator = device.makeCommandAllocator() else {
                    throw RendererError.makeAllocatorFailed
                }
                return allocator
            }

            sharedEvent = device.makeSharedEvent()
            sharedEvent?.signaledValue = 0

            compiler = try device.makeCompiler(descriptor: .init())
            if let metalLayer = mtkView.layer as? CAMetalLayer {
                queue.addResidencySet(metalLayer.residencySet)
            }
        } catch {
            drawingEnabled = false
            isReady = false
            print("RenderCoordinator setup failed: \(error)")
            return
        }

        reloadEncoder()
    }

    func setModelKind(_ kind: ModelKind) {
        guard modelKind != kind else { return }
        modelKind = kind
        reloadEncoder()
    }

    func setRenderMode(_ mode: RenderMode) {
        renderMode = mode
    }

    func draw(in view: MTKView) {
        guard drawingEnabled,
              let device,
              let commandQueue,
              let encoder,
              let drawable = view.currentDrawable else {
            return
        }

        print("[RenderCoordinator] draw frame \(frameNumber) for model \(modelKind.rawValue) mode \(renderMode.rawValue)")

        if frameNumber >= UInt64(maxFramesInFlight) {
            let waitValue = frameNumber - UInt64(maxFramesInFlight)
            sharedEvent?.wait(untilSignaledValue: waitValue, timeoutMS: 10)
        }

        let allocator = commandAllocators[Int(frameNumber % UInt64(maxFramesInFlight))]
        allocator.reset()

        guard let commandBuffer = device.makeCommandBuffer() else {
            print("RenderCoordinator: makeCommandBuffer failed")
            return
        }
        commandBuffer.beginCommandBuffer(allocator: allocator)
        let desired = renderMode
        let actualMode = encoder.supports(desired) ? desired : .perPixel
        encoder.encode(drawableTexture: drawable.texture, commandBuffer: commandBuffer, mode: actualMode)
        commandBuffer.endCommandBuffer()

        commandQueue.waitForDrawable(drawable)
        commandQueue.commit([commandBuffer])
        commandQueue.signalDrawable(drawable)
        drawable.present()

        if let sharedEvent {
            commandQueue.signalEvent(sharedEvent, value: frameNumber)
        }

        frameNumber += 1
    }

    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {}
}

private extension RenderCoordinator {
    enum RendererError: Error {
        case makeQueueFailed
        case makeAllocatorFailed
    }

    func reloadEncoder() {
        guard let device, let commandQueue, let compiler else {
            encoder = nil
            isReady = false
            return
        }

        do {
            let library = try device.makeDefaultLibrary(bundle: .main)
            switch modelKind {
            case .siren:
                encoder = try SirenEncoder(device: device, library: library, compiler: compiler, queue: commandQueue)
                loadMetadata(resourceName: "siren")
                if imageSize == .zero { updateImageDimensions(width: 512, height: 512) }
            case .fourier:
                encoder = try FourierEncoder(device: device, library: library, compiler: compiler, queue: commandQueue)
                loadMetadata(resourceName: "fourier")
                if imageSize == .zero { updateImageDimensions(width: 512, height: 512) }
            case .instantNGP:
                guard let model = loadInstantNGPModel() else {
                    encoder = nil
                    drawingEnabled = false
                    isReady = false
                    print("[RenderCoordinator] Missing instant_ngp model")
                    return
                }
                let weights = try model.makeMetalWeights(device: device)
                encoder = try InstantNGPEncoder(device: device, library: library, compiler: compiler, queue: commandQueue, weights: weights)
                if let img = model.metadata?.image,
                   let width = img.width,
                   let height = img.height {
                    updateImageDimensions(width: width, height: height)
                } else {
                    updateImageDimensions(width: 512, height: 512)
                }
            }
            print("[RenderCoordinator] Loaded model \(modelKind.rawValue) size=\(imageSize)")
        } catch {
            encoder = nil
            drawingEnabled = false
            isReady = false
            print("RenderCoordinator failed to load encoder for \(modelKind): \(error)")
            return
        }

        drawingEnabled = encoder != nil
        isReady = drawingEnabled
    }

    func updateImageDimensions(width: Int, height: Int) {
        let clampedWidth = max(width, 1)
        let clampedHeight = max(height, 1)
        imageSize = CGSize(width: clampedWidth, height: clampedHeight)
        aspectRatio = CGFloat(clampedWidth) / CGFloat(clampedHeight)
        guard let view = metalView else { return }
        view.drawableSize = imageSize
#if os(macOS)
        view.setNeedsDisplay(view.bounds)
#else
        view.setNeedsDisplay()
#endif
    }

    func loadMetadata(resourceName: String) {
        guard let url = Bundle.main.url(forResource: resourceName, withExtension: "json"),
              let data = try? Data(contentsOf: url),
              let model = try? JSONDecoder().decode(ModelFile.self, from: data),
              let image = model.metadata?.image,
              let width = image.width,
              let height = image.height else {
            print("[RenderCoordinator] No metadata for \(resourceName), using default size")
            return
        }
        updateImageDimensions(width: width, height: height)
        print("[RenderCoordinator] Metadata for \(resourceName): \(width)x\(height)")
    }

    func loadInstantNGPModel() -> InstantNGPModel? {
        guard let url = Bundle.main.url(forResource: "instant_ngp", withExtension: "json") else {
            print("instant_ngp.json not found in bundle")
            return nil
        }
        do {
            return try InstantNGPModel.load(from: url)
        } catch {
            print("Failed to load instant_ngp model: \(error)")
            return nil
        }
    }
}
#endif

private extension RenderCoordinator {
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
}
