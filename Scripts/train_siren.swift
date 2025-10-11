import Foundation
import Metal
import MetalKit
import QuartzCore
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers
import simd

#if !TRAINING_CLI
#if canImport(MetalTensorOp)
import MetalTensorOp
#endif

#if canImport(MetalTensorOp)
typealias SirenMetadata = MetalTensorOp.Metadata
typealias SirenModelDescriptor = MetalTensorOp.ModelDescriptor
typealias SirenModelSample = MetalTensorOp.ModelSample
typealias SirenMLP = MetalTensorOp.MLP
#else
struct ModelDescriptor: Codable {
    let type: String?
    init(type: String? = nil) {
        self.type = type
    }
}

struct Metadata: Codable {
    let mode: String?
    struct ImageMetadata: Codable {
        let width: Int?
        let height: Int?
        let aspect_ratio: Float?
    }
    struct SDFMetadata: Codable {
        let resolution: Int?
    }
    let image: ImageMetadata?
    let sdf: SDFMetadata?

    init(mode: String? = nil, image: ImageMetadata? = nil, sdf: SDFMetadata? = nil) {
        self.mode = mode
        self.image = image
        self.sdf = sdf
    }
}

struct ModelSample: Codable {
    let position: [Float]
    let value: [Float]
}

typealias SirenMetadata = Metadata
typealias SirenModelDescriptor = ModelDescriptor
typealias SirenModelSample = ModelSample
typealias SirenMLP = MLP

struct SirenModel: Codable {
    var metadata: Metadata?
    var model: ModelDescriptor?
    var mlp: MLP?
    var sampleCount: Int?
    var sampleSeed: UInt64?
    var samples: [ModelSample]?

    init(metadata: Metadata? = nil, model: ModelDescriptor? = nil, mlp: MLP? = nil, sampleCount: Int? = nil, sampleSeed: UInt64? = nil, samples: [ModelSample]? = nil) {
        self.metadata = metadata
        self.model = model
        self.mlp = mlp
        self.sampleCount = sampleCount
        self.sampleSeed = sampleSeed
        self.samples = samples
    }
}
#endif
#endif
#if TRAINING_CLI
typealias SirenMetadata = Metadata
typealias SirenModelDescriptor = ModelDescriptor
typealias SirenModelSample = ModelSample
typealias SirenMLP = MLP
#endif

struct SirenWeightsExport: Codable {
    var metadata: SirenMetadata?
    var model: SirenModelDescriptor?
    var mlp: SirenMLP?
    var sampleCount: Int?
    var sampleSeed: UInt64?
    var samples: [SirenModelSample]?

    init(metadata: SirenMetadata?, model: SirenModelDescriptor?, mlp: SirenMLP?, sampleCount: Int?, sampleSeed: UInt64?, samples: [SirenModelSample]?) {
        self.metadata = metadata
        self.model = model
        self.mlp = mlp
        self.sampleCount = sampleCount
        self.sampleSeed = sampleSeed
        self.samples = samples
    }
}

struct CLIOptions {
    let inputURL: URL
    let outputWeightsURL: URL
    let outputImageURL: URL
    let steps: Int
    let logInterval: Int
    let evalInterval: Int
    let learningRate: Float
    let sampleLimit: Int?
    let trainBatchSize: Int

    static func parse() throws -> CLIOptions {
        struct ArgError: Error, CustomStringConvertible {
            let description: String
        }

        var inputPath: String?
        var weightsPath = "trained_siren.json"
        var imagePath = "trained_output.png"
        var steps = 500
        var logInterval = 50
        var evalInterval = 50
        var learningRate: Float = 1e-3
        var sampleLimit: Int?
        var trainBatchSize = 2048

        var iter = CommandLine.arguments.dropFirst().makeIterator()
        while let arg = iter.next() {
            switch arg {
            case "-i", "--input":
                guard let value = iter.next() else { throw ArgError(description: "Missing value for --input") }
                inputPath = value
            case "-w", "--weights":
                guard let value = iter.next() else { throw ArgError(description: "Missing value for --weights") }
                weightsPath = value
            case "-o", "--output-image":
                guard let value = iter.next() else { throw ArgError(description: "Missing value for --output-image") }
                imagePath = value
            case "-s", "--steps":
                guard let value = iter.next(), let intVal = Int(value) else {
                    throw ArgError(description: "Invalid integer for --steps")
                }
                steps = max(1, intVal)
            case "-l", "--log-interval":
                guard let value = iter.next(), let intVal = Int(value) else {
                    throw ArgError(description: "Invalid integer for --log-interval")
                }
                logInterval = max(1, intVal)
            case "-e", "--eval-interval":
                guard let value = iter.next(), let intVal = Int(value) else {
                    throw ArgError(description: "Invalid integer for --eval-interval")
                }
                evalInterval = max(1, intVal)
            case "--lr", "--learning-rate":
                guard let value = iter.next(), let floatVal = Float(value) else {
                    throw ArgError(description: "Invalid float for --learning-rate")
                }
                learningRate = max(1e-6, floatVal)
            case "--limit":
                guard let value = iter.next(), let intVal = Int(value) else {
                    throw ArgError(description: "Invalid integer for --limit")
                }
                sampleLimit = max(1, intVal)
            case "--train-batch":
                guard let value = iter.next(), let intVal = Int(value) else {
                    throw ArgError(description: "Invalid integer for --train-batch")
                }
                trainBatchSize = max(1, intVal)
            case "-h", "--help":
                throw ArgError(description: usage())
            default:
                throw ArgError(description: "Unknown argument: \(arg)\n\n" + usage())
            }
        }

        guard let inputPath else {
            throw ArgError(description: "Missing --input path\n\n" + usage())
        }

        let inputURL = URL(fileURLWithPath: inputPath)
        let weightsURL = URL(fileURLWithPath: weightsPath)
        let imageURL = URL(fileURLWithPath: imagePath)

        return CLIOptions(
            inputURL: inputURL,
            outputWeightsURL: weightsURL,
            outputImageURL: imageURL,
            steps: steps,
            logInterval: logInterval,
            evalInterval: evalInterval,
            learningRate: learningRate,
            sampleLimit: sampleLimit,
            trainBatchSize: trainBatchSize
        )
    }

    static func usage() -> String {
        """
        Usage: train_siren --input <image> [options]

        Options:
          -i, --input <path>           Input image path (required)
          -w, --weights <path>         Output JSON weights path (default: trained_siren.json)
          -o, --output-image <path>    Output reconstructed PNG path (default: trained_output.png)
          -s, --steps <int>            Optimizer steps (default: 500)
          -l, --log-interval <int>     Print loss every N steps (default: 50)
          -e, --eval-interval <int>    Evaluate PSNR every N steps (default: 50)
          --lr, --learning-rate <f>    Adam learning rate (default: 1e-3)
          --limit <int>               Limit number of dataset samples
          --train-batch <int>         Number of samples per optimizer step (default: 2048)
          -h, --help                   Show this help message
        """
    }
}

struct Dataset {
    let positions: [SIMD2<Float>]
    let targets: [SIMD3<Float>]
    let width: Int
    let height: Int
}

@main
struct TrainSirenCLI {
    static func main() throws {
        let options = try CLIOptions.parse()

        let image = try loadImage(from: options.inputURL)
        var dataset = makeDataset(from: image)
        if let limit = options.sampleLimit, limit < dataset.positions.count {
            let limitedPositions = Array(dataset.positions.prefix(limit))
            let limitedTargets = Array(dataset.targets.prefix(limit))
            dataset = Dataset(positions: limitedPositions, targets: limitedTargets, width: limit, height: 1)
        }

        let device = try requireDevice()
        let compiler = try device.makeCompiler(descriptor: .init())
        guard let queue = device.makeMTL4CommandQueue() else {
            throw RuntimeError("Failed to create MTL4 command queue")
        }
        guard let allocator = device.makeCommandAllocator() else {
            throw RuntimeError("Failed to create command allocator")
        }

        let library = try makeTrainingLibrary(device: device)

        let trainingEngine = try SirenTrainingEngine(
            device: device,
            library: library,
            compiler: compiler,
            commandQueue: queue,
            maxFramesInFlight: 1
        )
        trainingEngine.hyperParameters.learningRate = options.learningRate
        try trainingEngine.loadDataset(image: image)
        if let limit = options.sampleLimit {
            trainingEngine.limitDataset(to: limit)
        }
        let totalSamples = max(trainingEngine.datasetSampleCount(), 1)
        if options.trainBatchSize > trainingEngine.maxChunkedBatchSize() {
            print("[train_siren] Requested train batch \(options.trainBatchSize) exceeds chunk capacity; clamping to \(trainingEngine.maxChunkedBatchSize())")
        }
        let requestedBatch = min(options.trainBatchSize, totalSamples)
        trainingEngine.setTrainingBatchSampleCount(requestedBatch)
        trainingEngine.setSamplingMode(.randomWithReplacement)

        var datasetSampleCount = trainingEngine.datasetSampleCount()
        if datasetSampleCount == 0 {
            datasetSampleCount = dataset.positions.count
        }
        var latestLoss: Float = .nan

        var bestLoss: Float = .greatestFiniteMagnitude
        var bestPSNR: Double = 0
        var bestImage: [UInt8] = []
        var bestDimensions = (width: dataset.width, height: dataset.height)

        var accumulatedStepMS: Double = 0

        for step in 0..<options.steps {
            guard let commandBuffer = device.makeCommandBuffer() else {
                throw RuntimeError("Failed to create command buffer")
            }
            let stepStart = CFAbsoluteTimeGetCurrent()
            commandBuffer.beginCommandBuffer(allocator: allocator)
            trainingEngine.encodeTrainingStep(frameIndex: step, commandBuffer: commandBuffer)
            commandBuffer.endCommandBuffer()
            let commitOptions = MTL4CommitOptions()
            let completionSemaphore = DispatchSemaphore(value: 0)
            var commitError: Error?
            commitOptions.addFeedbackHandler { feedback in
                if let error = feedback.error {
                    commitError = error
                }
                completionSemaphore.signal()
            }
            queue.commit([commandBuffer], options: commitOptions)
            completionSemaphore.wait()
            if let commitError {
                throw RuntimeError("GPU execution failed: \(commitError.localizedDescription)")
            }
            accumulatedStepMS += (CFAbsoluteTimeGetCurrent() - stepStart) * 1000.0

            guard let meanLoss = trainingEngine.takeLoss(for: step) else {
                throw RuntimeError("Loss value unavailable for frame \(step)")
            }
            latestLoss = meanLoss

            if step % options.logInterval == 0 || step + 1 == options.steps {
                print(String(format: "step %4d  loss %.6f", step + 1, latestLoss))
            }

            if (step + 1) % options.evalInterval == 0 || step + 1 == options.steps {
                let (pixels, psnr, _, mse, width, height) = try evaluateModel(
                    engine: trainingEngine,
                    queue: queue,
                    allocator: allocator
                )
                print(String(format: "          psnr %.2f dB  mse %.6f", psnr, mse))
                if latestLoss < bestLoss {
                    bestLoss = latestLoss
                    bestPSNR = psnr
                    bestImage = pixels
                    bestDimensions = (width, height)
                }
            }
        }

        if bestImage.isEmpty {
            let (pixels, psnr, sse, mse, width, height) = try evaluateModel(
                engine: trainingEngine,
                queue: queue,
                allocator: allocator
            )
            bestImage = pixels
            bestPSNR = psnr
            bestLoss = latestLoss
            bestDimensions = (width, height)
            print(String(format: "Evaluation summary: sse %.6f mse %.6f psnr %.2f", sse, mse, psnr))
        }

        if let layer = trainingEngine.mlp.layers.first {
            let dims = layer.weightTensor.dimensions.extents
            let rows = dims[0]
            let cols = dims[1]
            let count = min(8, rows * cols)
            var sampleWeights = [Float16](repeating: 0, count: count)
            let sliceRows = min(rows, count)
            let strides = MTLTensorExtents([1, sliceRows])!
            let sliceOrigin = MTLTensorExtents([0, 0])!
            let sliceDimensions = MTLTensorExtents([sliceRows, min(cols, 1)])!
            layer.weightTensor.getBytes(
                &sampleWeights,
                strides: strides,
                sliceOrigin: sliceOrigin,
                sliceDimensions: sliceDimensions
            )
            let floats = sampleWeights.map { Float($0) }
            print("Sample weights", floats)
        }

        if let firstPixel = bestImage.prefix(4).map({ $0 }) as [UInt8]? {
            print("Sample pixel RGBA:", firstPixel)
        }

        try saveImage(pixels: bestImage, width: bestDimensions.width, height: bestDimensions.height, to: options.outputImageURL)
        try saveWeights(mlp: trainingEngine.mlp, to: options.outputWeightsURL)

        print(String(format: "Finished. Best loss %.6f, PSNR %.2f dB", bestLoss, bestPSNR))
        print("Weights saved to \(options.outputWeightsURL.path)")
        print("Reconstruction saved to \(options.outputImageURL.path)")
        let avgStepMS = accumulatedStepMS / Double(max(options.steps, 1))
        print(String(format: "Average training step %.3f ms", avgStepMS))
    }

    private static func requireDevice() throws -> MTLDevice {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw RuntimeError("Metal is not supported on this device")
        }
        guard device.supportsFamily(.metal4) else {
            throw RuntimeError("GPU does not support Metal 4 tensor operations")
        }
        return device
    }

    private static func loadImage(from url: URL) throws -> CGImage {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil) else {
            throw RuntimeError("Failed to open image at \(url.path)")
        }
        guard let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
            throw RuntimeError("Failed to decode image at \(url.path)")
        }
        return image
    }

    private static func makeDataset(from image: CGImage) -> Dataset {
        let width = image.width
        let height = image.height
        let count = width * height

        let colorSpace = CGColorSpace(name: CGColorSpace.sRGB)!
        var pixels = [UInt8](repeating: 0, count: count * 4)
        let context = CGContext(
            data: &pixels,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: width * 4,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        )!
        let rect = CGRect(x: 0, y: 0, width: width, height: height)
        context.draw(image, in: rect)

        var positions = [SIMD2<Float>](repeating: .zero, count: count)
        var targets = [SIMD3<Float>](repeating: .zero, count: count)

        for y in 0..<height {
            for x in 0..<width {
                let index = y * width + x
                let pixelIndex = index * 4
                let r = Float(pixels[pixelIndex + 0]) / 255.0
                let g = Float(pixels[pixelIndex + 1]) / 255.0
                let b = Float(pixels[pixelIndex + 2]) / 255.0
                targets[index] = SIMD3<Float>(r, g, b)

                let nx = (Float(x) + 0.5) / Float(max(width, 1))
                let ny = (Float(y) + 0.5) / Float(max(height, 1))
                positions[index] = SIMD2<Float>(nx * 2.0 - 1.0, ny * 2.0 - 1.0)
            }
        }

        return Dataset(positions: positions, targets: targets, width: width, height: height)
    }

    private static func makeTrainingLibrary(device: MTLDevice) throws -> MTLLibrary {
        let fileManager = FileManager.default
        let cwd = URL(fileURLWithPath: fileManager.currentDirectoryPath)
        let shaderDir = cwd.appendingPathComponent("MetalTensorOp/Shaders", isDirectory: true)

        func load(name: String) throws -> String {
            let url = shaderDir.appendingPathComponent(name)
            return try String(contentsOf: url, encoding: .utf8)
        }

        let common = try load(name: "MLPCommon.metal")
        let training = try load(name: "SirenTraining.metal").replacingOccurrences(of: "#include \"MLPCommon.metal\"", with: common)
        let inference = try load(name: "SirenMLP.metal").replacingOccurrences(of: "#include \"MLPCommon.metal\"", with: common)

        let source = common + "\n" + training + "\n" + inference

        let options = MTLCompileOptions()
        do {
            return try device.makeLibrary(source: source, options: options)
        } catch {
            fputs("Shader compilation failed: \(error)\n", stderr)
            throw error
        }
    }

    private static func evaluateModel(
        engine: SirenTrainingEngine,
        queue: MTL4CommandQueue,
        allocator: MTL4CommandAllocator
    ) throws -> ([UInt8], Double, Float, Float, Int, Int) {
        let result = try engine.evaluateDataset(queue: queue, allocator: allocator)
        return (result.rgba, Double(result.psnr), result.sse, result.mse, result.width, result.height)
    }

    private static func saveImage(pixels: [UInt8], width: Int, height: Int, to url: URL) throws {
        let pixelCount = pixels.count / 4
        var outputWidth = max(width, 1)
        var outputHeight = max(height, 1)
        if outputWidth * outputHeight != pixelCount && pixelCount > 0 {
            outputWidth = max(outputWidth, Int(sqrt(Double(pixelCount))))
            outputWidth = max(1, min(pixelCount, outputWidth))
            outputHeight = max(1, pixelCount / outputWidth)
            if outputWidth * outputHeight < pixelCount {
                outputHeight = pixelCount / outputWidth + ((pixelCount % outputWidth) == 0 ? 0 : 1)
            }
        }

        let colorSpace = CGColorSpace(name: CGColorSpace.sRGB)!
        let dataProvider = CGDataProvider(data: Data(pixels) as CFData)!
        let cgImage = CGImage(
            width: outputWidth,
            height: outputHeight,
            bitsPerComponent: 8,
            bitsPerPixel: 32,
            bytesPerRow: outputWidth * 4,
            space: colorSpace,
            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue),
            provider: dataProvider,
            decode: nil,
            shouldInterpolate: false,
            intent: .defaultIntent
        )!

        guard let destination = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil) else {
            throw RuntimeError("Failed to create image destination at \(url.path)")
        }
        CGImageDestinationAddImage(destination, cgImage, nil)
        CGImageDestinationFinalize(destination)
    }

    private static func saveWeights(mlp: MLP, to url: URL) throws {
        let sirenModel = SirenWeightsExport(metadata: nil, model: nil, mlp: mlp, sampleCount: nil, sampleSeed: nil, samples: nil)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(sirenModel)
        try data.write(to: url)
    }

}

struct RuntimeError: Error, CustomStringConvertible {
    let description: String
    init(_ description: String) { self.description = description }
}
