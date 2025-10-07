import SwiftUI
import Combine
import Metal
import MetalKit

final class SirenTrainingCoordinator: NSObject, MTKViewDelegate, ObservableObject {
    @Published var loss: Float = 0
    @Published var isTraining: Bool = false
    @Published var groundTruthImage: CGImage?

    private(set) var device: MTLDevice?
    private var commandQueue: MTL4CommandQueue?
    private var commandAllocators: [MTL4CommandAllocator] = []
    private var sharedEvent: MTLSharedEvent?
    private var frameNumber: UInt64 = 0
    private let maxFramesInFlight = 3

    private var engine: SirenTrainingEngine?
    private weak var metalView: MTKView?

    private var trainingDimensions: CGSize = .init(width: 512, height: 512)

    var aspectRatio: CGFloat {
        guard trainingDimensions.height > 0 else { return 1 }
        return trainingDimensions.width / trainingDimensions.height
    }

    func setup(device: MTLDevice, mtkView: MTKView) {
        self.device = device
        self.metalView = mtkView

        do {
            guard let queue = device.makeMTL4CommandQueue() else {
                print("Failed to create MTL4CommandQueue")
                return
            }
            commandQueue = queue

            commandAllocators = []
            for _ in 0..<maxFramesInFlight {
                guard let allocator = device.makeCommandAllocator() else {
                    print("Failed to create command allocator")
                    return
                }
                commandAllocators.append(allocator)
            }

            sharedEvent = device.makeSharedEvent()
            sharedEvent?.signaledValue = 0

            let compiler = try device.makeCompiler(descriptor: .init())
            let library = try device.makeDefaultLibrary(bundle: .main)

            engine = try SirenTrainingEngine(
                device: device,
                library: library,
                compiler: compiler,
                commandQueue: queue,
                maxFramesInFlight: maxFramesInFlight
            )

            engine?.lossHandler = { [weak self] loss in
                DispatchQueue.main.async {
                    self?.loss = loss
                }
            }

            if let metalLayer = mtkView.layer as? CAMetalLayer {
                queue.addResidencySet(metalLayer.residencySet)
            }
            mtkView.drawableSize = trainingDimensions
            updateDrawableSizes()
        } catch {
            print("Coordinator setup error: \(error)")
        }
    }

    func loadImage(url: URL) {
        guard let engine else { return }
        #if os(macOS)
        guard let imageSource = CGImageSourceCreateWithURL(url as CFURL, nil),
              let image = CGImageSourceCreateImageAtIndex(imageSource, 0, nil) else {
            print("Failed to load image at \(url)")
            return
        }
        #else
        guard let data = try? Data(contentsOf: url),
              let uiImage = UIImage(data: data),
              let cgImage = uiImage.cgImage else {
            print("Failed to load image at \(url)")
            return
        }
        let image = cgImage
        #endif

        do {
            try engine.loadDataset(image: image)
            try engine.resetWeights()
            let total = engine.datasetSampleCount()
            let minBatch = 32
            let maxChunk = engine.maxChunkedBatchSize()
            let suggestedTarget = max(minBatch, min(total / 8, total))
            let suggested = min(maxChunk, suggestedTarget)
            engine.setTrainingBatchSampleCount(suggested)
            DispatchQueue.main.async {
                self.isTraining = false
                self.groundTruthImage = image
                self.trainingDimensions = CGSize(width: image.width, height: image.height)
                self.updateDrawableSizes()
                self.loss = 0
                self.frameNumber = 0
            }
        } catch {
            print("Dataset load error: \(error)")
        }
    }

    func toggleTraining() {
        guard groundTruthImage != nil else { return }
        isTraining.toggle()
    }

    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        updateDrawableSizes()
    }

    func updateDrawableSizes() {
        guard trainingDimensions.width > 0, trainingDimensions.height > 0 else { return }
        guard let engine else { return }
        engine.updateRenderDimensions(width: Int(trainingDimensions.width), height: Int(trainingDimensions.height))
    }

    func draw(in view: MTKView) {
        guard let device,
              let commandQueue,
              let engine else {
            return
        }

        if frameNumber >= UInt64(maxFramesInFlight) {
            let waitValue = frameNumber - UInt64(maxFramesInFlight)
            sharedEvent?.wait(untilSignaledValue: waitValue, timeoutMS: 10)
        }

        guard let drawable = view.currentDrawable else {
            return
        }

        let frameIndex = Int(frameNumber % UInt64(maxFramesInFlight))
        let allocator = commandAllocators[frameIndex]
        allocator.reset()

        guard let commandBuffer = device.makeCommandBuffer() else { return }
        commandBuffer.beginCommandBuffer(allocator: allocator)

        if isTraining {
            engine.encodeTrainingStep(frameIndex: frameIndex, commandBuffer: commandBuffer)
        }

        engine.encodeRender(to: drawable.texture, commandBuffer: commandBuffer)

        commandBuffer.endCommandBuffer()
        commandQueue.waitForDrawable(drawable)
        if isTraining {
            let commitOptions = MTL4CommitOptions()
            let completedIndex = frameIndex
            commitOptions.addFeedbackHandler { feedback in
                guard feedback.error == nil else { return }
                engine.handleCompletedFrame(completedIndex)
            }
            commandQueue.commit([commandBuffer], options: commitOptions)
        } else {
            commandQueue.commit([commandBuffer])
        }
        commandQueue.signalDrawable(drawable)
        drawable.present()

        if let sharedEvent {
            commandQueue.signalEvent(sharedEvent, value: frameNumber)
        }

        frameNumber += 1
    }
}
