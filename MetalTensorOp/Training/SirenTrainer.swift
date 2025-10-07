import Foundation
import Metal
import MetalKit
import simd

private let sirenTrainBatchSize = 32
private let sirenTrainMaxDim = 64
private let sirenTrainInputDim = 2
private let sirenTrainOutputDim = 3
private let sirenTrainMaxChunks = 64 // Supports up to 2048-sample batches

struct SirenTrainingParamsUniform {
    var batchStart: UInt32
    var totalSamples: UInt32
    var sliceBatchSize: UInt32
    var layerCount: UInt32
    var globalBatchSize: UInt32
}

struct SirenAdamParamsUniform {
    var layerCount: UInt32
    var totalWeightCount: UInt32
    var totalBiasCount: UInt32
    var learningRate: Float
    var beta1: Float
    var beta2: Float
    var epsilon: Float
    var beta1Denom: Float
    var beta2Denom: Float
    var weightDecay: Float
    var preserveGradients: UInt32
    var padding: UInt32 = 0
}

private struct TrainingFrameResources {
    let paramsBuffer: MTLBuffer
    let lossBuffer: MTLBuffer
    let activationHistoryBuffer: MTLBuffer
    let preActivationHistoryBuffer: MTLBuffer
}

private struct EvaluationResources {
    let outputsBuffer: MTLBuffer
    let diffSquaresBuffer: MTLBuffer
    let resultBuffer: MTLBuffer
    let rgbaBuffer: MTLBuffer
    let sampleCountBuffer: MTLBuffer
    let threadgroupCount: Int
    let sampleCount: Int
}

final class SirenTrainingEngine {
    struct HyperParameters {
        var learningRate: Float = 1e-3
        var beta1: Float = 0.9
        var beta2: Float = 0.999
        var epsilon: Float = 1e-8
        var weightDecay: Float = 0.0
    }

    struct EvaluationResult {
        let sse: Float
        let mse: Float
        let psnr: Float
        let rgba: [UInt8]
        let width: Int
        let height: Int
    }

    let device: MTLDevice
    let library: MTLLibrary
    let compiler: MTL4Compiler
    let commandQueue: MTL4CommandQueue

    let trainingPipeline: MTLComputePipelineState
    let trainingArgumentTable: any MTL4ArgumentTable
    let adamWeightsPipeline: MTLComputePipelineState
    let adamBiasesPipeline: MTLComputePipelineState
    let adamArgumentTable: any MTL4ArgumentTable
    let trainingResidencySet: MTLResidencySet
    let trainingFence: MTLFence?
    let evalPipeline: MTLComputePipelineState
    let reducePsnrPipeline: MTLComputePipelineState
    let packOutputsPipeline: MTLComputePipelineState
    let evalArgumentTable: any MTL4ArgumentTable
    let packArgumentTable: any MTL4ArgumentTable
    let reduceArgumentTable: any MTL4ArgumentTable

    private(set) var mlp: MLP
    let sirenEncoder: SirenEncoder

    private var datasetWidth: Int = 0
    private var datasetHeight: Int = 0
    private var originalPositions: [SIMD2<Float>] = []
    private var originalTargets: [SIMD3<Float>] = []

    private let layerInputDimsBuffer: MTLBuffer
    private let layerOutputDimsBuffer: MTLBuffer
    private let gradientTensors: [MTLTensor]
    private let gradientBuffers: [MTLBuffer]
    private let gradientArgumentsBuffer: MTLBuffer
    private let momentArgumentsBuffer1: MTLBuffer
    private let momentArgumentsBuffer2: MTLBuffer
    private let momentLayers1: [MTLTensor]
    private let momentLayers2: [MTLTensor]
    private let momentBuffers1: [MTLBuffer]
    private let momentBuffers2: [MTLBuffer]
    private let adamParamsBuffer: MTLBuffer
    private let totalWeightCount: Int
    private let totalBiasCount: Int

    private let frameResources: [TrainingFrameResources]
    private let maxFramesInFlight: Int
    private struct PendingLoss {
        let batchCount: Int
        let lossBuffer: MTLBuffer
    }
    private var pendingLoss: [PendingLoss?]
    private(set) var positionsBuffer: MTLBuffer?
    private(set) var targetsBuffer: MTLBuffer?

    private var totalSamples: Int = 0
    private var batchCursor: Int = 0
    private var step: UInt64 = 0

    var hyperParameters = HyperParameters()
    var lossHandler: ((Float) -> Void)?

    private var evaluationResources: EvaluationResources?
    private var hostPositions: [SIMD2<Float>] = []
    private var hostTargets: [SIMD3<Float>] = []
    private var needsHostShuffle = false
    private var trainingBatchSampleCount: Int = 1024

    init(
        device: MTLDevice,
        library: MTLLibrary,
        compiler: MTL4Compiler,
        commandQueue: MTL4CommandQueue,
        maxFramesInFlight: Int = 3
    ) throws {
        self.device = device
        self.library = library
        self.compiler = compiler
        self.commandQueue = commandQueue
        self.maxFramesInFlight = max(1, maxFramesInFlight)

        self.mlp = try SirenTrainingEngine.makeRandomMLP(device: device)
        self.sirenEncoder = try SirenEncoder(
            device: device,
            library: library,
            compiler: compiler,
            queue: commandQueue,
            mlp: mlp,
            metadata: nil,
            trainingDimensions: nil
        )
        let dims = try SirenTrainingEngine.makeLayerDimensionBuffers(device: device, mlp: mlp)
        self.layerInputDimsBuffer = dims.input
        self.layerOutputDimsBuffer = dims.output

        let gradientResources = try SirenTrainingEngine.makeGradientTensors(device: device, mlp: mlp)
        self.gradientTensors = gradientResources.tensors
        self.gradientBuffers = gradientResources.buffers
        self.gradientArgumentsBuffer = gradientResources.argumentBuffer

        let trainFunction = MTL4LibraryFunctionDescriptor()
        trainFunction.name = "sirenTrainStep"
        trainFunction.library = library

        let trainPipelineDescriptor = MTL4ComputePipelineDescriptor()
        trainPipelineDescriptor.computeFunctionDescriptor = trainFunction
        self.trainingPipeline = try compiler.makeComputePipelineState(descriptor: trainPipelineDescriptor)

        let trainingTableDesc = MTL4ArgumentTableDescriptor()
        trainingTableDesc.maxBufferBindCount = 10
        self.trainingArgumentTable = try device.makeArgumentTable(descriptor: trainingTableDesc)
        self.trainingFence = device.makeFence()

        let momentResources = try SirenTrainingEngine.makeMomentTensors(device: device, mlp: mlp)
        self.momentLayers1 = momentResources.firstMoments
        self.momentLayers2 = momentResources.secondMoments
        self.momentBuffers1 = momentResources.firstBuffers
        self.momentBuffers2 = momentResources.secondBuffers
        self.momentArgumentsBuffer1 = momentResources.arguments1
        self.momentArgumentsBuffer2 = momentResources.arguments2

        let adamWeightsFunction = MTL4LibraryFunctionDescriptor()
        adamWeightsFunction.name = "sirenAdamUpdateWeights"
        adamWeightsFunction.library = library
        let adamWeightsDescriptor = MTL4ComputePipelineDescriptor()
        adamWeightsDescriptor.computeFunctionDescriptor = adamWeightsFunction
        self.adamWeightsPipeline = try compiler.makeComputePipelineState(descriptor: adamWeightsDescriptor)

        let adamBiasesFunction = MTL4LibraryFunctionDescriptor()
        adamBiasesFunction.name = "sirenAdamUpdateBiases"
        adamBiasesFunction.library = library
        let adamBiasesDescriptor = MTL4ComputePipelineDescriptor()
        adamBiasesDescriptor.computeFunctionDescriptor = adamBiasesFunction
        self.adamBiasesPipeline = try compiler.makeComputePipelineState(descriptor: adamBiasesDescriptor)

        let evalFunction = MTL4LibraryFunctionDescriptor()
        evalFunction.name = "sirenEvalCoopBuffer"
        evalFunction.library = library
        let evalDescriptor = MTL4ComputePipelineDescriptor()
        evalDescriptor.computeFunctionDescriptor = evalFunction
        self.evalPipeline = try compiler.makeComputePipelineState(descriptor: evalDescriptor)

        let reduceFunction = MTL4LibraryFunctionDescriptor()
        reduceFunction.name = "sirenReduceDiffSquares"
        reduceFunction.library = library
        let reduceDescriptor = MTL4ComputePipelineDescriptor()
        reduceDescriptor.computeFunctionDescriptor = reduceFunction
        self.reducePsnrPipeline = try compiler.makeComputePipelineState(descriptor: reduceDescriptor)

        let packFunction = MTL4LibraryFunctionDescriptor()
        packFunction.name = "sirenPackOutputsRGBA"
        packFunction.library = library
        let packDescriptor = MTL4ComputePipelineDescriptor()
        packDescriptor.computeFunctionDescriptor = packFunction
        self.packOutputsPipeline = try compiler.makeComputePipelineState(descriptor: packDescriptor)

        let evalTableDesc = MTL4ArgumentTableDescriptor()
        evalTableDesc.maxBufferBindCount = 7
        self.evalArgumentTable = try device.makeArgumentTable(descriptor: evalTableDesc)
        evalArgumentTable.setAddress(sirenEncoder.tensorArgumentsBuffer.gpuAddress, index: 0)
        evalArgumentTable.setAddress(sirenEncoder.layerCountBuffer.gpuAddress, index: 1)

        let packTableDesc = MTL4ArgumentTableDescriptor()
        packTableDesc.maxBufferBindCount = 3
        self.packArgumentTable = try device.makeArgumentTable(descriptor: packTableDesc)

        let reduceTableDesc = MTL4ArgumentTableDescriptor()
        reduceTableDesc.maxBufferBindCount = 3
        self.reduceArgumentTable = try device.makeArgumentTable(descriptor: reduceTableDesc)

        let adamTableDesc = MTL4ArgumentTableDescriptor()
        adamTableDesc.maxBufferBindCount = 7
        self.adamArgumentTable = try device.makeArgumentTable(descriptor: adamTableDesc)

        guard let adamParamsBuffer = device.makeBuffer(length: MemoryLayout<SirenAdamParamsUniform>.stride, options: .storageModeShared) else {
            throw SirenEncoderError.failedToCreateBuffer
        }
        self.adamParamsBuffer = adamParamsBuffer

        self.pendingLoss = Array(repeating: nil, count: self.maxFramesInFlight)

        self.totalWeightCount = mlp.layers.reduce(0) { partial, layer in
            let dims = layer.weightTensor.dimensions.extents
            return partial + dims[0] * dims[1]
        }
        self.totalBiasCount = mlp.layers.reduce(0) { partial, layer in
            let dims = layer.biasTensor.dimensions.extents
            return partial + dims[0]
        }

        trainingArgumentTable.setAddress(sirenEncoder.tensorArgumentsBuffer.gpuAddress, index: 0)
        trainingArgumentTable.setAddress(gradientArgumentsBuffer.gpuAddress, index: 1)
        trainingArgumentTable.setAddress(layerInputDimsBuffer.gpuAddress, index: 5)
        trainingArgumentTable.setAddress(layerOutputDimsBuffer.gpuAddress, index: 6)

        adamArgumentTable.setAddress(sirenEncoder.tensorArgumentsBuffer.gpuAddress, index: 0)
        adamArgumentTable.setAddress(gradientArgumentsBuffer.gpuAddress, index: 1)
        adamArgumentTable.setAddress(momentArgumentsBuffer1.gpuAddress, index: 2)
        adamArgumentTable.setAddress(momentArgumentsBuffer2.gpuAddress, index: 3)
        adamArgumentTable.setAddress(adamParamsBuffer.gpuAddress, index: 4)
        adamArgumentTable.setAddress(layerInputDimsBuffer.gpuAddress, index: 5)
        adamArgumentTable.setAddress(layerOutputDimsBuffer.gpuAddress, index: 6)

        self.frameResources = try SirenTrainingEngine.makeFrameResources(
            device: device,
            layerCount: mlp.layers.count,
            frameCount: self.maxFramesInFlight
        )

        let residency = try device.makeResidencySet(descriptor: .init())
        commandQueue.addResidencySet(residency)
        residency.addAllocation(sirenEncoder.tensorArgumentsBuffer)
        residency.addAllocation(sirenEncoder.layerCountBuffer)
        residency.addAllocation(sirenEncoder.renderUniformsBuffer)
        residency.addAllocation(sirenEncoder.numSamplesBuffer)
        residency.addAllocation(layerInputDimsBuffer)
        residency.addAllocation(layerOutputDimsBuffer)
        residency.addAllocation(gradientArgumentsBuffer)
        residency.addAllocation(momentArgumentsBuffer1)
        residency.addAllocation(momentArgumentsBuffer2)
        residency.addAllocation(adamParamsBuffer)
        for frame in frameResources {
            residency.addAllocation(frame.paramsBuffer)
            residency.addAllocation(frame.lossBuffer)
            residency.addAllocation(frame.activationHistoryBuffer)
            residency.addAllocation(frame.preActivationHistoryBuffer)
        }
        for buffer in gradientBuffers {
            residency.addAllocation(buffer)
        }
        for layer in momentLayers1 {
            residency.addAllocation(layer)
            if let buffer = layer as? MTLBuffer { residency.addAllocation(buffer) }
        }
        for layer in momentLayers2 {
            residency.addAllocation(layer)
            if let buffer = layer as? MTLBuffer { residency.addAllocation(buffer) }
        }
        residency.commit()
        self.trainingResidencySet = residency
    }

    func resetWeights() throws {
        try SirenTrainingEngine.randomizeWeights(device: device, layers: mlp.layers)
        zeroMoments()
    }

    func loadDataset(image: CGImage) throws {
        let (positions, targets) = try SirenTrainingEngine.makeDataset(from: image)
        try loadDataset(positions: positions, targets: targets, width: image.width, height: image.height)
    }

    func loadDataset(positions: [SIMD2<Float>], targets: [SIMD3<Float>], width: Int, height: Int) throws {
        precondition(positions.count == targets.count, "positions/targets mismatch")
        hostPositions = positions
        hostTargets = targets
        totalSamples = positions.count
        datasetWidth = width
        datasetHeight = height
        originalPositions = positions
        originalTargets = targets
        batchCursor = 0
        step = 0
        needsHostShuffle = true

        trainingBatchSampleCount = min(trainingBatchSampleCount, totalSamples)

        try ensureDatasetBuffersCapacity(sampleCount: positions.count)
        uploadDatasetPrefix(totalSamples)
        shuffleDatasetIfNeeded()

        sirenEncoder.updateTrainingDimensions(width: width, height: height)
        sirenEncoder.setNumSamples(totalSamples)
        zeroMoments()

        try prepareEvaluationResources(sampleCount: totalSamples)
    }

    private func ensureDatasetBuffersCapacity(sampleCount: Int) throws {
        let positionLength = sampleCount * MemoryLayout<SIMD2<Float>>.stride
        let targetLength = sampleCount * MemoryLayout<SIMD3<Float>>.stride

        if positionsBuffer == nil || positionsBuffer!.length < positionLength {
            guard let buffer = device.makeBuffer(length: positionLength, options: .storageModeShared) else {
                throw SirenEncoderError.failedToCreateBuffer
            }
            positionsBuffer = buffer
            trainingResidencySet.addAllocation(buffer)
        }

        if targetsBuffer == nil || targetsBuffer!.length < targetLength {
            guard let buffer = device.makeBuffer(length: targetLength, options: .storageModeShared) else {
                throw SirenEncoderError.failedToCreateBuffer
            }
            targetsBuffer = buffer
            trainingResidencySet.addAllocation(buffer)
        }

        trainingResidencySet.commit()
    }

    private func uploadDatasetPrefix(_ count: Int) {
        copyDatasetToGPU(positions: hostPositions, targets: hostTargets, count: count)
    }

    private func copyDatasetToGPU(positions: [SIMD2<Float>], targets: [SIMD3<Float>], count: Int) {
        guard count > 0 else { return }
        guard let positionsBuffer, let targetsBuffer else { return }
        precondition(positions.count >= count && targets.count >= count, "dataset copy count exceeds source length")
        let positionLength = count * MemoryLayout<SIMD2<Float>>.stride
        let targetLength = count * MemoryLayout<SIMD3<Float>>.stride
        _ = positions.withUnsafeBytes { src in
            memcpy(positionsBuffer.contents(), src.baseAddress!, positionLength)
        }
        _ = targets.withUnsafeBytes { src in
            memcpy(targetsBuffer.contents(), src.baseAddress!, targetLength)
        }
    }

    private func shuffleDatasetIfNeeded() {
        guard needsHostShuffle else { return }
        defer { needsHostShuffle = false }
        guard totalSamples > 1 else {
            uploadDatasetPrefix(totalSamples)
            return
        }
        var rng = SystemRandomNumberGenerator()
        for i in stride(from: totalSamples - 1, through: 1, by: -1) {
            let j = Int.random(in: 0...i, using: &rng)
            if i != j {
                hostPositions.swapAt(i, j)
                hostTargets.swapAt(i, j)
            }
        }
        uploadDatasetPrefix(totalSamples)
    }

    func limitDataset(to sampleLimit: Int) {
        guard let positionsBuffer else { return }
        let maxSamples = positionsBuffer.length / MemoryLayout<SIMD2<Float>>.stride
        let newTotal = max(1, min(sampleLimit, maxSamples))
        totalSamples = newTotal
        datasetWidth = newTotal
        datasetHeight = 1
        batchCursor = min(batchCursor, newTotal - 1)
        sirenEncoder.setNumSamples(newTotal)
        needsHostShuffle = true
        shuffleDatasetIfNeeded()
        trainingBatchSampleCount = min(trainingBatchSampleCount, newTotal)
        try? prepareEvaluationResources(sampleCount: newTotal)
        if originalPositions.count > newTotal {
            originalPositions = Array(originalPositions.prefix(newTotal))
            originalTargets = Array(originalTargets.prefix(newTotal))
        }
    }

    func encodeTrainingStep(frameIndex: Int, commandBuffer: MTL4CommandBuffer) {
        guard let positionsBuffer, let targetsBuffer, totalSamples > 0 else { return }

        zeroGradientBuffers()
        shuffleDatasetIfNeeded()

        let datasetRemaining = max(totalSamples - batchCursor, 0)
        if datasetRemaining <= 0 {
            batchCursor = 0
            return
        }

        let desiredBatch = min(trainingBatchSampleCount, datasetRemaining)
        if desiredBatch <= 0 {
            return
        }

        let batchStart = batchCursor
        let frame = frameResources[frameIndex % frameResources.count]

        var params = SirenTrainingParamsUniform(
            batchStart: UInt32(batchStart),
            totalSamples: UInt32(totalSamples),
            sliceBatchSize: 0,
            layerCount: UInt32(mlp.layers.count),
            globalBatchSize: UInt32(desiredBatch)
        )
        frame.lossBuffer.contents().bindMemory(to: Float.self, capacity: 1).pointee = 0

        let beta1Pow = pow(hyperParameters.beta1, Float(step + 1))
        let beta2Pow = pow(hyperParameters.beta2, Float(step + 1))
        let beta1Denom = max(1.0 - beta1Pow, 1e-8)
        let beta2Denom = max(1.0 - beta2Pow, 1e-8)

        var adamParams = SirenAdamParamsUniform(
            layerCount: UInt32(mlp.layers.count),
            totalWeightCount: UInt32(totalWeightCount),
            totalBiasCount: UInt32(totalBiasCount),
            learningRate: hyperParameters.learningRate,
            beta1: hyperParameters.beta1,
            beta2: hyperParameters.beta2,
            epsilon: hyperParameters.epsilon,
            beta1Denom: Float(beta1Denom),
            beta2Denom: Float(beta2Denom),
            weightDecay: hyperParameters.weightDecay,
            preserveGradients: 0,
            padding: 0
        )
        adamParamsBuffer.contents().copyMemory(from: &adamParams, byteCount: MemoryLayout<SirenAdamParamsUniform>.stride)

        trainingArgumentTable.setAddress(positionsBuffer.gpuAddress, index: 2)
        trainingArgumentTable.setAddress(targetsBuffer.gpuAddress, index: 3)
        trainingArgumentTable.setAddress(frame.paramsBuffer.gpuAddress, index: 4)
        trainingArgumentTable.setAddress(layerInputDimsBuffer.gpuAddress, index: 5)
        trainingArgumentTable.setAddress(layerOutputDimsBuffer.gpuAddress, index: 6)
        trainingArgumentTable.setAddress(frame.activationHistoryBuffer.gpuAddress, index: 7)
        trainingArgumentTable.setAddress(frame.preActivationHistoryBuffer.gpuAddress, index: 8)
        trainingArgumentTable.setAddress(frame.lossBuffer.gpuAddress, index: 9)

        let maxSliceSamples = sirenTrainBatchSize * sirenTrainMaxChunks
        let threadsPerThreadgroup = MTLSize(width: trainingPipeline.threadExecutionWidth * 4, height: 1, depth: 1)
        var processedSamples = 0

        while processedSamples < desiredBatch {
            let sliceCount = min(desiredBatch - processedSamples, maxSliceSamples)
            params.batchStart = UInt32(batchStart + processedSamples)
            params.sliceBatchSize = UInt32(sliceCount)
            frame.paramsBuffer.contents().copyMemory(from: &params, byteCount: MemoryLayout<SirenTrainingParamsUniform>.stride)

            let chunkCount = max(1, (sliceCount + sirenTrainBatchSize - 1) / sirenTrainBatchSize)

            guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
            commandBuffer.useResidencySet(trainingResidencySet)
            encoder.setComputePipelineState(trainingPipeline)
            encoder.setArgumentTable(trainingArgumentTable)
            encoder.dispatchThreadgroups(
                threadgroupsPerGrid: MTLSize(width: chunkCount, height: 1, depth: 1),
                threadsPerThreadgroup: threadsPerThreadgroup
            )
            if let fence = trainingFence {
                encoder.updateFence(fence, afterEncoderStages: .dispatch)
            }
            encoder.endEncoding()

            processedSamples += sliceCount
        }

        if totalWeightCount > 0 {
            guard let weightEncoder = commandBuffer.makeComputeCommandEncoder() else { return }
            commandBuffer.useResidencySet(trainingResidencySet)
            if let fence = trainingFence {
                weightEncoder.waitForFence(fence, beforeEncoderStages: .dispatch)
            }
            weightEncoder.setComputePipelineState(adamWeightsPipeline)
            weightEncoder.setArgumentTable(adamArgumentTable)
            let threadsPerGroup = max(adamWeightsPipeline.threadExecutionWidth, 1)
            let threadgroups = (totalWeightCount + threadsPerGroup - 1) / threadsPerGroup
            weightEncoder.dispatchThreadgroups(
                threadgroupsPerGrid: MTLSize(width: threadgroups, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: threadsPerGroup, height: 1, depth: 1)
            )
            if let fence = trainingFence {
                weightEncoder.updateFence(fence, afterEncoderStages: .dispatch)
            }
            weightEncoder.endEncoding()
        }

        if totalBiasCount > 0 {
            guard let biasEncoder = commandBuffer.makeComputeCommandEncoder() else { return }
            commandBuffer.useResidencySet(trainingResidencySet)
            if let fence = trainingFence {
                biasEncoder.waitForFence(fence, beforeEncoderStages: .dispatch)
            }
            biasEncoder.setComputePipelineState(adamBiasesPipeline)
            biasEncoder.setArgumentTable(adamArgumentTable)
            let threadsPerGroup = max(adamBiasesPipeline.threadExecutionWidth, 1)
            let threadgroups = (totalBiasCount + threadsPerGroup - 1) / threadsPerGroup
            biasEncoder.dispatchThreadgroups(
                threadgroupsPerGrid: MTLSize(width: threadgroups, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: threadsPerGroup, height: 1, depth: 1)
            )
            if let fence = trainingFence {
                biasEncoder.updateFence(fence, afterEncoderStages: .dispatch)
            }
            biasEncoder.endEncoding()
        }

        let batchForStep = desiredBatch
        let slot = frameIndex % pendingLoss.count
        pendingLoss[slot] = PendingLoss(batchCount: batchForStep, lossBuffer: frame.lossBuffer)


        step += 1
        batchCursor += desiredBatch
        if batchCursor >= totalSamples {
            batchCursor = 0
            needsHostShuffle = true
        }
    }

    func encodeRender(to texture: MTLTexture, commandBuffer: MTL4CommandBuffer) {
        sirenEncoder.encode(drawableTexture: texture, commandBuffer: commandBuffer, mode: .cooperative)
    }

    func updateRenderDimensions(width: Int, height: Int) {
        sirenEncoder.updateTrainingDimensions(width: width, height: height)
    }

    func setTrainingBatchSampleCount(_ count: Int) {
        trainingBatchSampleCount = max(sirenTrainBatchSize, count)
    }

    func handleCompletedFrame(_ frameIndex: Int) {
        guard let meanLoss = takeLoss(for: frameIndex) else {
            return
        }
        DispatchQueue.main.async { [weak self] in
            self?.lossHandler?(meanLoss)
        }
    }

    private func zeroMoments() {
        for buffer in gradientBuffers {
            memset(buffer.contents(), 0, buffer.length)
        }
        for buffer in momentBuffers1 {
            memset(buffer.contents(), 0, buffer.length)
        }
        for buffer in momentBuffers2 {
            memset(buffer.contents(), 0, buffer.length)
        }
    }

    private func prepareEvaluationResources(sampleCount: Int) throws {
        let threadgroupCount = (sampleCount + sirenTrainBatchSize - 1) / sirenTrainBatchSize

        let outputsLength = sampleCount * MemoryLayout<SIMD3<Float>>.stride
        let diffSquaresLength = sampleCount * MemoryLayout<Float>.stride
        let resultLength = 3 * MemoryLayout<Float>.stride
        let rgbaLength = sampleCount * MemoryLayout<UInt32>.stride

        guard let outputsBuffer = device.makeBuffer(length: outputsLength, options: .storageModeShared),
              let diffSquaresBuffer = device.makeBuffer(length: diffSquaresLength, options: .storageModeShared),
              let resultBuffer = device.makeBuffer(length: resultLength, options: .storageModeShared),
              let rgbaBuffer = device.makeBuffer(length: rgbaLength, options: .storageModeShared),
              let sampleCountBuffer = device.makeBuffer(length: MemoryLayout<UInt32>.stride, options: .storageModeShared) else {
            throw SirenEncoderError.failedToCreateBuffer
        }

        memset(diffSquaresBuffer.contents(), 0, diffSquaresLength)
        memset(resultBuffer.contents(), 0, resultLength)
        var sampleCountValue = UInt32(sampleCount)
        memcpy(sampleCountBuffer.contents(), &sampleCountValue, MemoryLayout<UInt32>.stride)

        trainingResidencySet.addAllocation(outputsBuffer)
        trainingResidencySet.addAllocation(diffSquaresBuffer)
        trainingResidencySet.addAllocation(resultBuffer)
        trainingResidencySet.addAllocation(rgbaBuffer)
        trainingResidencySet.addAllocation(sampleCountBuffer)
        trainingResidencySet.commit()

        evaluationResources = EvaluationResources(
            outputsBuffer: outputsBuffer,
            diffSquaresBuffer: diffSquaresBuffer,
            resultBuffer: resultBuffer,
            rgbaBuffer: rgbaBuffer,
            sampleCountBuffer: sampleCountBuffer,
            threadgroupCount: threadgroupCount,
            sampleCount: sampleCount
        )
    }

    func evaluateDataset(queue: MTL4CommandQueue, allocator: MTL4CommandAllocator) throws -> EvaluationResult {
        guard let positionsBuffer, let targetsBuffer else {
            throw SirenEncoderError.failedToCreateBuffer
        }
        guard let eval = evaluationResources else {
            throw SirenEncoderError.failedToCreateBuffer
        }

        copyDatasetToGPU(positions: originalPositions, targets: originalTargets, count: originalPositions.count)

        guard let commandBuffer = device.makeCommandBuffer() else {
            throw SirenEncoderError.failedToCreateBuffer
        }
        commandBuffer.beginCommandBuffer(allocator: allocator)

        var sampleCountValue = UInt32(eval.sampleCount)
        memcpy(eval.sampleCountBuffer.contents(), &sampleCountValue, MemoryLayout<UInt32>.stride)

        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            commandBuffer.useResidencySet(trainingResidencySet)
            encoder.setComputePipelineState(evalPipeline)
            evalArgumentTable.setAddress(positionsBuffer.gpuAddress, index: 2)
            evalArgumentTable.setAddress(targetsBuffer.gpuAddress, index: 3)
            evalArgumentTable.setAddress(eval.outputsBuffer.gpuAddress, index: 4)
            evalArgumentTable.setAddress(eval.diffSquaresBuffer.gpuAddress, index: 5)
            evalArgumentTable.setAddress(eval.sampleCountBuffer.gpuAddress, index: 6)
            encoder.setArgumentTable(evalArgumentTable)

            let threadsPerThreadgroup = MTLSize(
                width: evalPipeline.threadExecutionWidth * 4,
                height: 1,
                depth: 1
            )
            let threadgroups = MTLSize(width: eval.threadgroupCount, height: 1, depth: 1)
            encoder.dispatchThreadgroups(
                threadgroupsPerGrid: threadgroups,
                threadsPerThreadgroup: threadsPerThreadgroup
            )
            encoder.endEncoding()
        }

        if let packEncoder = commandBuffer.makeComputeCommandEncoder() {
            commandBuffer.useResidencySet(trainingResidencySet)
            packEncoder.setComputePipelineState(packOutputsPipeline)
            packArgumentTable.setAddress(eval.outputsBuffer.gpuAddress, index: 0)
            packArgumentTable.setAddress(eval.rgbaBuffer.gpuAddress, index: 1)
            packArgumentTable.setAddress(eval.sampleCountBuffer.gpuAddress, index: 2)
            packEncoder.setArgumentTable(packArgumentTable)

            let threadsPerThreadgroup = MTLSize(width: packOutputsPipeline.threadExecutionWidth, height: 1, depth: 1)
            let threadgroups = MTLSize(
                width: (eval.sampleCount + threadsPerThreadgroup.width - 1) / threadsPerThreadgroup.width,
                height: 1,
                depth: 1
            )
            packEncoder.dispatchThreadgroups(
                threadgroupsPerGrid: threadgroups,
                threadsPerThreadgroup: threadsPerThreadgroup
            )
            packEncoder.endEncoding()
        }

        if let reduceEncoder = commandBuffer.makeComputeCommandEncoder() {
            commandBuffer.useResidencySet(trainingResidencySet)
            reduceEncoder.setComputePipelineState(reducePsnrPipeline)
            reduceArgumentTable.setAddress(eval.diffSquaresBuffer.gpuAddress, index: 0)
            reduceArgumentTable.setAddress(eval.resultBuffer.gpuAddress, index: 1)
            reduceArgumentTable.setAddress(eval.sampleCountBuffer.gpuAddress, index: 2)
            reduceEncoder.setArgumentTable(reduceArgumentTable)
            reduceEncoder.dispatchThreadgroups(
                threadgroupsPerGrid: MTLSize(width: 1, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1)
            )
            reduceEncoder.endEncoding()
        }

        commandBuffer.endCommandBuffer()

        let options = MTL4CommitOptions()
        let semaphore = DispatchSemaphore(value: 0)
        var commitError: Error?
        options.addFeedbackHandler { feedback in
            if let error = feedback.error {
                commitError = error
            }
            semaphore.signal()
        }
        queue.commit([commandBuffer], options: options)
        semaphore.wait()

        if let _ = commitError {
            throw SirenEncoderError.failedToCreateBuffer
        }

        let resultPointer = eval.resultBuffer.contents().bindMemory(to: Float.self, capacity: 3)
        let sse = resultPointer[0]
        let mse = resultPointer[1]
        let psnr = resultPointer[2]

        let rgbaCount = eval.sampleCount * 4
        let rgbaPointer = eval.rgbaBuffer.contents().bindMemory(to: UInt8.self, capacity: rgbaCount)
        let rgba = Array(UnsafeBufferPointer(start: rgbaPointer, count: rgbaCount))

        let dims = datasetDimensions()

        copyDatasetToGPU(positions: hostPositions, targets: hostTargets, count: totalSamples)

        return EvaluationResult(sse: sse, mse: mse, psnr: psnr, rgba: rgba, width: dims.width, height: dims.height)
    }

    func takeLoss(for frameIndex: Int) -> Float? {
        let slot = frameIndex % pendingLoss.count
        guard let pending = pendingLoss[slot] else {
            return nil
        }
        pendingLoss[slot] = nil
        let rawLoss = pending.lossBuffer.contents().bindMemory(to: Float.self, capacity: 1).pointee
        let channelCount = sirenTrainOutputDim
        let normalization = pending.batchCount * channelCount
        let meanLoss = rawLoss / Float(max(normalization, 1))
        return meanLoss
    }

    func datasetSampleCount() -> Int {
        return totalSamples
    }

    func maxChunkedBatchSize() -> Int {
        return sirenTrainBatchSize * sirenTrainMaxChunks
    }

    func datasetDimensions() -> (width: Int, height: Int) {
        return (max(datasetWidth, 1), max(datasetHeight, 1))
    }

    func zeroGradientBuffers() {
        for buffer in gradientBuffers {
            memset(buffer.contents(), 0, buffer.length)
        }
    }

}

private extension SirenTrainingEngine {
    static func makeRandomMLP(device: MTLDevice) throws -> MLP {
        let layerDimensions: [(input: Int, output: Int)] = [
            (sirenTrainInputDim, sirenTrainMaxDim),
            (sirenTrainMaxDim, sirenTrainMaxDim),
            (sirenTrainMaxDim, sirenTrainMaxDim),
            (sirenTrainMaxDim, sirenTrainMaxDim),
            (sirenTrainMaxDim, sirenTrainMaxDim),
            (sirenTrainMaxDim, sirenTrainMaxDim),
            (sirenTrainMaxDim, sirenTrainMaxDim),
            (sirenTrainMaxDim, sirenTrainMaxDim),
            (sirenTrainMaxDim, sirenTrainMaxDim),
            (sirenTrainMaxDim, sirenTrainOutputDim)
        ]

        var layers: [MLPParameterLayer] = []
        layers.reserveCapacity(layerDimensions.count)

        for (layerIndex, dims) in layerDimensions.enumerated() {
            let rows = dims.output
            let cols = dims.input

            guard let extents = MTLTensorExtents([rows, cols]) else {
                throw SirenEncoderError.failedToCreateBuffer
            }

            let descriptor = MTLTensorDescriptor()
            descriptor.dimensions = extents
            descriptor.strides = MTLTensorExtents([1, rows])
            descriptor.dataType = .float16
            descriptor.usage = .compute

            let count = rows * cols
            var weightValues = [Float16](repeating: 0, count: count)
            let scale: Float
            if layerIndex == 0 {
                scale = 1.0 / Float(cols)
            } else {
                scale = sqrt(6.0 / Float(cols))
            }
            for col in 0..<cols {
                for row in 0..<rows {
                    let idx = col * rows + row
                    weightValues[idx] = Float16(Float.random(in: -scale...scale))
                }
            }

            guard let weightBuffer = device.makeBuffer(bytes: weightValues, length: count * MemoryLayout<Float16>.stride, options: .storageModeShared) else {
                throw SirenEncoderError.failedToCreateBuffer
            }

            let weightTensor = try weightBuffer.makeTensor(descriptor: descriptor, offset: 0)

            guard let biasExtents = MTLTensorExtents([rows]) else {
                throw SirenEncoderError.failedToCreateBuffer
            }
            let biasDescriptor = MTLTensorDescriptor()
            biasDescriptor.dimensions = biasExtents
            biasDescriptor.strides = MTLTensorExtents([1])
            biasDescriptor.dataType = .float16
            biasDescriptor.usage = .compute

            var biasValues = [Float16](repeating: 0, count: rows)
            if layerIndex < layerDimensions.count - 1 {
                for row in 0..<rows {
                    biasValues[row] = Float16(Float.random(in: -scale...scale))
                }
            }

            guard let biasBuffer = device.makeBuffer(bytes: &biasValues, length: rows * MemoryLayout<Float16>.stride, options: .storageModeShared) else {
                throw SirenEncoderError.failedToCreateBuffer
            }
            let biasTensor = try biasBuffer.makeTensor(descriptor: biasDescriptor, offset: 0)

            layers.append(MLPParameterLayer(weightTensor: weightTensor, biasTensor: biasTensor))
        }

        return MLP(layers: layers)
    }

    static func randomizeWeights(device: MTLDevice, layers: [MLPParameterLayer]) throws {
        for (layerIndex, layer) in layers.enumerated() {
            guard let weightBuffer = layer.weightTensor as? MTLBuffer else { continue }
            let rows = layer.weightTensor.dimensions.extents[0]
            let cols = layer.weightTensor.dimensions.extents[1]
            let count = rows * cols
            let pointer = weightBuffer.contents().bindMemory(to: Float16.self, capacity: count)
            let scale: Float = layerIndex == 0 ? (1.0 / Float(cols)) : sqrt(6.0 / Float(cols))
            for col in 0..<cols {
                for row in 0..<rows {
                    let idx = col * rows + row
                    pointer[idx] = Float16(Float.random(in: -scale...scale))
                }
            }

            if let biasBuffer = layer.biasTensor as? MTLBuffer {
                let pointer = biasBuffer.contents().bindMemory(to: Float16.self, capacity: rows)
                if layerIndex == layers.count - 1 {
                    memset(biasBuffer.contents(), 0, biasBuffer.length)
                } else {
                    for row in 0..<rows {
                        pointer[row] = Float16(Float.random(in: -scale...scale))
                    }
                }
            }
        }
    }

    static func makeLayerDimensionBuffers(device: MTLDevice, mlp: MLP) throws -> (input: MTLBuffer, output: MTLBuffer) {
        var inputDims = [UInt32]()
        var outputDims = [UInt32]()
        inputDims.reserveCapacity(mlp.layers.count)
        outputDims.reserveCapacity(mlp.layers.count)

        for layer in mlp.layers {
            let dims = layer.weightTensor.dimensions.extents
            outputDims.append(UInt32(dims[0]))
            inputDims.append(UInt32(dims[1]))
        }

        guard let inputBuffer = device.makeBuffer(bytes: inputDims, length: inputDims.count * MemoryLayout<UInt32>.stride, options: .storageModeShared),
              let outputBuffer = device.makeBuffer(bytes: outputDims, length: outputDims.count * MemoryLayout<UInt32>.stride, options: .storageModeShared) else {
            throw SirenEncoderError.failedToCreateBuffer
        }

        return (inputBuffer, outputBuffer)
    }

    static func makeMomentTensors(device: MTLDevice, mlp: MLP) throws -> (firstMoments: [MTLTensor], firstBuffers: [MTLBuffer], secondMoments: [MTLTensor], secondBuffers: [MTLBuffer], arguments1: MTLBuffer, arguments2: MTLBuffer) {
        var firstMoments: [MTLTensor] = []
        var secondMoments: [MTLTensor] = []
        var firstBuffers: [MTLBuffer] = []
        var secondBuffers: [MTLBuffer] = []
        var args1 = MetalMLPTensorArguments()
        var args2 = MetalMLPTensorArguments()

        for (index, layer) in mlp.layers.enumerated() {
            let weightDims = layer.weightTensor.dimensions.extents
            guard let weightExtents = MTLTensorExtents([weightDims[0], weightDims[1]]) else {
                throw SirenEncoderError.failedToCreateBuffer
            }

            let descriptor = MTLTensorDescriptor()
            descriptor.dimensions = weightExtents
            descriptor.strides = MTLTensorExtents([1, weightDims[0]])
            descriptor.dataType = .float32
            descriptor.usage = .compute

            let count = weightDims[0] * weightDims[1]
            let length = count * MemoryLayout<Float>.stride

            guard let buffer1 = device.makeBuffer(length: length, options: .storageModeShared),
                  let buffer2 = device.makeBuffer(length: length, options: .storageModeShared) else {
                throw SirenEncoderError.failedToCreateBuffer
            }
            memset(buffer1.contents(), 0, length)
            memset(buffer2.contents(), 0, length)

            let tensor1 = try buffer1.makeTensor(descriptor: descriptor, offset: 0)
            let tensor2 = try buffer2.makeTensor(descriptor: descriptor, offset: 0)

            firstMoments.append(tensor1)
            secondMoments.append(tensor2)
            firstBuffers.append(buffer1)
            secondBuffers.append(buffer2)
            args1.weight[index] = tensor1.gpuResourceID
            args2.weight[index] = tensor2.gpuResourceID

            let biasDims = layer.biasTensor.dimensions.extents
            guard let biasExtents = MTLTensorExtents([biasDims[0]]) else {
                throw SirenEncoderError.failedToCreateBuffer
            }
            let biasDescriptor = MTLTensorDescriptor()
            biasDescriptor.dimensions = biasExtents
            biasDescriptor.strides = MTLTensorExtents([1])
            biasDescriptor.dataType = .float32
            biasDescriptor.usage = .compute

            let biasLength = biasDims[0] * MemoryLayout<Float>.stride
            guard let biasBuffer1 = device.makeBuffer(length: biasLength, options: .storageModeShared),
                  let biasBuffer2 = device.makeBuffer(length: biasLength, options: .storageModeShared) else {
                throw SirenEncoderError.failedToCreateBuffer
            }
            memset(biasBuffer1.contents(), 0, biasLength)
            memset(biasBuffer2.contents(), 0, biasLength)

            let biasTensor1 = try biasBuffer1.makeTensor(descriptor: biasDescriptor, offset: 0)
            let biasTensor2 = try biasBuffer2.makeTensor(descriptor: biasDescriptor, offset: 0)

            firstMoments.append(biasTensor1)
            secondMoments.append(biasTensor2)
            firstBuffers.append(biasBuffer1)
            secondBuffers.append(biasBuffer2)
            args1.bias[index] = biasTensor1.gpuResourceID
            args2.bias[index] = biasTensor2.gpuResourceID
        }

        guard let argsBuffer1 = device.makeBuffer(length: MemoryLayout<MetalMLPTensorArguments>.stride, options: .storageModeShared),
              let argsBuffer2 = device.makeBuffer(length: MemoryLayout<MetalMLPTensorArguments>.stride, options: .storageModeShared) else {
            throw SirenEncoderError.failedToCreateBuffer
        }
        var copyArgs1 = args1
        var copyArgs2 = args2
        memcpy(argsBuffer1.contents(), &copyArgs1, MemoryLayout<MetalMLPTensorArguments>.stride)
        memcpy(argsBuffer2.contents(), &copyArgs2, MemoryLayout<MetalMLPTensorArguments>.stride)

        return (firstMoments, firstBuffers, secondMoments, secondBuffers, argsBuffer1, argsBuffer2)
    }

    static func makeGradientTensors(device: MTLDevice, mlp: MLP) throws -> (tensors: [MTLTensor], buffers: [MTLBuffer], argumentBuffer: MTLBuffer) {
        var tensors: [MTLTensor] = []
        var buffers: [MTLBuffer] = []
        var args = MetalMLPTensorArguments()

        for (index, layer) in mlp.layers.enumerated() {
            let weightDims = layer.weightTensor.dimensions.extents
            guard let weightExtents = MTLTensorExtents([weightDims[0], weightDims[1]]) else {
                throw SirenEncoderError.failedToCreateBuffer
            }

            let descriptor = MTLTensorDescriptor()
            descriptor.dimensions = weightExtents
            descriptor.strides = MTLTensorExtents([1, weightDims[0]])
            descriptor.dataType = .float32
            descriptor.usage = .compute

            let count = weightDims[0] * weightDims[1]
            let length = count * MemoryLayout<Float>.stride

            guard let buffer = device.makeBuffer(length: length, options: .storageModeShared) else {
                throw SirenEncoderError.failedToCreateBuffer
            }
            memset(buffer.contents(), 0, length)

            let tensor = try buffer.makeTensor(descriptor: descriptor, offset: 0)
            tensors.append(tensor)
            buffers.append(buffer)
            args.weight[index] = tensor.gpuResourceID

            let biasDims = layer.biasTensor.dimensions.extents
            guard let biasExtents = MTLTensorExtents([biasDims[0]]) else {
                throw SirenEncoderError.failedToCreateBuffer
            }
            let biasDescriptor = MTLTensorDescriptor()
            biasDescriptor.dimensions = biasExtents
            biasDescriptor.strides = MTLTensorExtents([1])
            biasDescriptor.dataType = .float32
            biasDescriptor.usage = .compute

            let biasLength = biasDims[0] * MemoryLayout<Float>.stride
            guard let biasBuffer = device.makeBuffer(length: biasLength, options: .storageModeShared) else {
                throw SirenEncoderError.failedToCreateBuffer
            }
            memset(biasBuffer.contents(), 0, biasLength)

            let biasTensor = try biasBuffer.makeTensor(descriptor: biasDescriptor, offset: 0)
            tensors.append(biasTensor)
            buffers.append(biasBuffer)
            args.bias[index] = biasTensor.gpuResourceID
        }

        guard let argsBuffer = device.makeBuffer(length: MemoryLayout<MetalMLPTensorArguments>.stride, options: .storageModeShared) else {
            throw SirenEncoderError.failedToCreateBuffer
        }

        var argsCopy = args
        memcpy(argsBuffer.contents(), &argsCopy, MemoryLayout<MetalMLPTensorArguments>.stride)

        return (tensors, buffers, argsBuffer)
    }

    static func makeFrameResources(device: MTLDevice, layerCount: Int, frameCount: Int) throws -> [TrainingFrameResources] {
        let chunkCapacity = sirenTrainMaxChunks
        let activationCount = (layerCount + 1) * sirenTrainBatchSize * sirenTrainMaxDim * chunkCapacity

        let activationLength = activationCount * MemoryLayout<Float16>.stride
        let preActivationLength = layerCount * sirenTrainBatchSize * sirenTrainMaxDim * chunkCapacity * MemoryLayout<Float16>.stride

        var frames: [TrainingFrameResources] = []
        frames.reserveCapacity(frameCount)

        for _ in 0..<frameCount {
            guard let paramsBuffer = device.makeBuffer(length: MemoryLayout<SirenTrainingParamsUniform>.stride, options: .storageModeShared),
                  let lossBuffer = device.makeBuffer(length: MemoryLayout<Float>.stride, options: .storageModeShared),
                  let activationBuffer = device.makeBuffer(length: activationLength, options: .storageModeShared),
                  let preActivationBuffer = device.makeBuffer(length: preActivationLength, options: .storageModeShared) else {
                throw SirenEncoderError.failedToCreateBuffer
            }

            frames.append(
                TrainingFrameResources(
                    paramsBuffer: paramsBuffer,
                    lossBuffer: lossBuffer,
                    activationHistoryBuffer: activationBuffer,
                    preActivationHistoryBuffer: preActivationBuffer
                )
            )
        }

        return frames
    }

    static func makeDataset(from image: CGImage) throws -> ([SIMD2<Float>], [SIMD3<Float>]) {
        let width = image.width
        let height = image.height
        let count = width * height

        guard let colorSpace = CGColorSpace(name: CGColorSpace.sRGB) else {
            throw SirenEncoderError.failedToCreateBuffer
        }

        var pixels = [UInt8](repeating: 0, count: count * 4)
        guard let context = CGContext(
            data: &pixels,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: width * 4,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else {
            throw SirenEncoderError.failedToCreateBuffer
        }

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

        return (positions, targets)
    }
}
