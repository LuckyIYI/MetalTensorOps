import Metal
import Foundation
import QuartzCore

enum InstantNGPError: Error {
    case failedToCreateBuffer
    case failedToAllocateHashTable
    case invalidConfiguration
    case failedToCreatePipeline
    case missingTensorStorage
}

struct InstantNGPConfig {
    static let numLevels = 16
    static let featuresPerLevel = 2
    static let log2HashmapSize = 12
    static let baseResolution = 16
    static let maxResolution = 2048
    static let totalFeatures = numLevels * featuresPerLevel

    static let mlpHiddenWidth = 64
    static let mlpOutputDim = 3
    static let mlpNumLayers = 2

    static let batchSize = 64
}

final class InstantNGPEncoder: ComputeEncoder {
    private let device: MTLDevice
    private let queue: MTL4CommandQueue

    private let renderPipelineState: MTLComputePipelineState
    private let inferencePipelineState: MTLComputePipelineState
    private let inferenceDebugPipelineState: MTLComputePipelineState

    private let renderArgumentTable: any MTL4ArgumentTable
    private let inferenceArgumentTable: any MTL4ArgumentTable
    private let inferenceDebugArgumentTable: any MTL4ArgumentTable

    private let renderResidencySet: MTLResidencySet
    private let inferenceResidencySet: MTLResidencySet

    private let weights: InstantNGPMetalWeights
    private let renderUniformsBuffer: MTLBuffer
    private let numPositionsBuffer: MTLBuffer

    struct RenderUniforms {
        var time: Float
        var trainingWidth: UInt32
        var trainingHeight: UInt32
        var padding: UInt32 = 0
    }


    init(
        device: MTLDevice,
        library: MTLLibrary,
        compiler: MTL4Compiler,
        queue: MTL4CommandQueue,
        weights: InstantNGPMetalWeights
    ) throws {
        self.device = device
        self.queue = queue

        self.weights = weights

        guard let renderUniformsBuffer = device.makeBuffer(length: MemoryLayout<RenderUniforms>.stride, options: .storageModeShared) else {
            throw InstantNGPError.failedToCreateBuffer
        }
        var initialUniforms = RenderUniforms(
            time: 0,
            trainingWidth: UInt32(max(weights.imageWidth, 1)),
            trainingHeight: UInt32(max(weights.imageHeight, 1))
        )
        memcpy(renderUniformsBuffer.contents(), &initialUniforms, MemoryLayout<RenderUniforms>.stride)
        self.renderUniformsBuffer = renderUniformsBuffer

        guard let numPositionsBuffer = device.makeBuffer(length: MemoryLayout<UInt32>.stride, options: .storageModeShared) else {
            throw InstantNGPError.failedToCreateBuffer
        }
        self.numPositionsBuffer = numPositionsBuffer

        let renderFn = MTL4LibraryFunctionDescriptor()
        renderFn.name = "instantNGPRender"
        renderFn.library = library
        let renderDesc = MTL4ComputePipelineDescriptor()
        renderDesc.computeFunctionDescriptor = renderFn
        self.renderPipelineState = try compiler.makeComputePipelineState(descriptor: renderDesc)

        let inferenceFn = MTL4LibraryFunctionDescriptor()
        inferenceFn.name = "instantNGPInference"
        inferenceFn.library = library
        let inferenceDesc = MTL4ComputePipelineDescriptor()
        inferenceDesc.computeFunctionDescriptor = inferenceFn
        self.inferencePipelineState = try compiler.makeComputePipelineState(descriptor: inferenceDesc)

        let debugFn = MTL4LibraryFunctionDescriptor()
        debugFn.name = "instantNGPInferenceDebug"
        debugFn.library = library
        let debugDesc = MTL4ComputePipelineDescriptor()
        debugDesc.computeFunctionDescriptor = debugFn
        self.inferenceDebugPipelineState = try compiler.makeComputePipelineState(descriptor: debugDesc)

        let renderTableDesc = MTL4ArgumentTableDescriptor()
        renderTableDesc.maxTextureBindCount = 1
        renderTableDesc.maxBufferBindCount = 6
        self.renderArgumentTable = try device.makeArgumentTable(descriptor: renderTableDesc)
        renderArgumentTable.setResource(weights.layer1Weights.gpuResourceID, bufferIndex: 0)
        renderArgumentTable.setResource(weights.layer1Bias.gpuResourceID, bufferIndex: 1)
        renderArgumentTable.setResource(weights.layer2Weights.gpuResourceID, bufferIndex: 2)
        renderArgumentTable.setResource(weights.layer2Bias.gpuResourceID, bufferIndex: 3)
        renderArgumentTable.setAddress(weights.hashTable.gpuAddress, index: 4)
        renderArgumentTable.setAddress(renderUniformsBuffer.gpuAddress, index: 5)

        let inferenceTableDesc = MTL4ArgumentTableDescriptor()
        inferenceTableDesc.maxBufferBindCount = 8
        self.inferenceArgumentTable = try device.makeArgumentTable(descriptor: inferenceTableDesc)
        inferenceArgumentTable.setResource(weights.layer1Weights.gpuResourceID, bufferIndex: 0)
        inferenceArgumentTable.setResource(weights.layer1Bias.gpuResourceID, bufferIndex: 1)
        inferenceArgumentTable.setResource(weights.layer2Weights.gpuResourceID, bufferIndex: 2)
        inferenceArgumentTable.setResource(weights.layer2Bias.gpuResourceID, bufferIndex: 3)
        inferenceArgumentTable.setAddress(weights.hashTable.gpuAddress, index: 4)
        inferenceArgumentTable.setAddress(numPositionsBuffer.gpuAddress, index: 7)

        let debugTableDesc = MTL4ArgumentTableDescriptor()
        debugTableDesc.maxBufferBindCount = 11
        self.inferenceDebugArgumentTable = try device.makeArgumentTable(descriptor: debugTableDesc)
        inferenceDebugArgumentTable.setResource(weights.layer1Weights.gpuResourceID, bufferIndex: 0)
        inferenceDebugArgumentTable.setResource(weights.layer1Bias.gpuResourceID, bufferIndex: 1)
        inferenceDebugArgumentTable.setResource(weights.layer2Weights.gpuResourceID, bufferIndex: 2)
        inferenceDebugArgumentTable.setResource(weights.layer2Bias.gpuResourceID, bufferIndex: 3)
        inferenceDebugArgumentTable.setAddress(weights.hashTable.gpuAddress, index: 4)
        inferenceDebugArgumentTable.setAddress(numPositionsBuffer.gpuAddress, index: 7)

        let renderResidency = try device.makeResidencySet(descriptor: .init())
        queue.addResidencySet(renderResidency)
        renderResidency.addAllocation(weights.hashTable)
        let addTensor: (MTLTensor, MTLResidencySet) -> Void = { tensor, residency in
            residency.addAllocation(tensor)
        }
        addTensor(weights.layer1Weights, renderResidency)
        addTensor(weights.layer1Bias, renderResidency)
        addTensor(weights.layer2Weights, renderResidency)
        addTensor(weights.layer2Bias, renderResidency)
        renderResidency.addAllocation(renderUniformsBuffer)
        renderResidency.commit()
        self.renderResidencySet = renderResidency

        let inferenceResidency = try device.makeResidencySet(descriptor: .init())
        queue.addResidencySet(inferenceResidency)
        inferenceResidency.addAllocation(weights.hashTable)
        addTensor(weights.layer1Weights, inferenceResidency)
        addTensor(weights.layer1Bias, inferenceResidency)
        addTensor(weights.layer2Weights, inferenceResidency)
        addTensor(weights.layer2Bias, inferenceResidency)
        inferenceResidency.addAllocation(numPositionsBuffer)
        inferenceResidency.commit()
        self.inferenceResidencySet = inferenceResidency
    }


    func encode(drawableTexture: MTLTexture, commandBuffer: MTL4CommandBuffer) {
        let pointer = renderUniformsBuffer.contents().bindMemory(to: RenderUniforms.self, capacity: 1)
        pointer.pointee.time = Float(CACurrentMediaTime())

        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        commandBuffer.useResidencySet(renderResidencySet)
        encoder.setComputePipelineState(renderPipelineState)
        encoder.setArgumentTable(renderArgumentTable)
        renderArgumentTable.setTexture(drawableTexture.gpuResourceID, index: 0)

        let gridSize = MTLSize(width: drawableTexture.width, height: drawableTexture.height, depth: 1)
        guard gridSize.width > 0 && gridSize.height > 0 else {
            encoder.endEncoding()
            return
        }

        let threadsPerThreadgroup = MTLSize(
            width: 16,
            height: 16,
            depth: 1
        )

        encoder.dispatchThreads(threadsPerGrid: gridSize, threadsPerThreadgroup: threadsPerThreadgroup)
        encoder.endEncoding()
    }

    func encodeInference(
        positions: MTLTensor,
        outputs: MTLTensor,
        numPositions: UInt32,
        commandBuffer: MTL4CommandBuffer
    ) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        commandBuffer.useResidencySet(inferenceResidencySet)
        encoder.setComputePipelineState(inferencePipelineState)
        encoder.setArgumentTable(inferenceArgumentTable)
        inferenceArgumentTable.setResource(positions.gpuResourceID, bufferIndex: 5)
        inferenceArgumentTable.setResource(outputs.gpuResourceID, bufferIndex: 6)
        var count = numPositions
        memcpy(numPositionsBuffer.contents(), &count, MemoryLayout<UInt32>.stride)

        let threadsPerThreadgroup = MTLSize(width: inferencePipelineState.threadExecutionWidth * 4, height: 1, depth: 1)
        let numBatches = max(1, (Int(numPositions) + InstantNGPConfig.batchSize - 1) / InstantNGPConfig.batchSize)
        let threadgroupsPerGrid = MTLSize(width: 1, height: numBatches, depth: 1)
        encoder.dispatchThreadgroups(
            threadgroupsPerGrid: threadgroupsPerGrid,
            threadsPerThreadgroup: threadsPerThreadgroup
        )
        encoder.endEncoding()
    }

    func encodeInferenceDebug(
        positions: MTLTensor,
        outputs: MTLTensor,
        numPositions: UInt32,
        encodedDebug: MTLTensor,
        hiddenDebug: MTLTensor,
        outputDebug: MTLTensor,
        commandBuffer: MTL4CommandBuffer
    ) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        commandBuffer.useResidencySet(inferenceResidencySet)
        encoder.setComputePipelineState(inferenceDebugPipelineState)
        encoder.setArgumentTable(inferenceDebugArgumentTable)
        inferenceDebugArgumentTable.setResource(positions.gpuResourceID, bufferIndex: 5)
        inferenceDebugArgumentTable.setResource(outputs.gpuResourceID, bufferIndex: 6)
        inferenceDebugArgumentTable.setResource(encodedDebug.gpuResourceID, bufferIndex: 8)
        inferenceDebugArgumentTable.setResource(hiddenDebug.gpuResourceID, bufferIndex: 9)
        inferenceDebugArgumentTable.setResource(outputDebug.gpuResourceID, bufferIndex: 10)
        var count = numPositions
        memcpy(numPositionsBuffer.contents(), &count, MemoryLayout<UInt32>.stride)

        let threadsPerThreadgroup = MTLSize(width: inferenceDebugPipelineState.threadExecutionWidth * 4, height: 1, depth: 1)
        let numBatches = max(1, (Int(numPositions) + InstantNGPConfig.batchSize - 1) / InstantNGPConfig.batchSize)
        let threadgroupsPerGrid = MTLSize(width: 1, height: numBatches, depth: 1)
        encoder.dispatchThreadgroups(
            threadgroupsPerGrid: threadgroupsPerGrid,
            threadsPerThreadgroup: threadsPerThreadgroup
        )
        encoder.endEncoding()
    }
}
