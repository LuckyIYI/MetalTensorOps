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
    private let renderCoopPipelineState: MTLComputePipelineState
    private let cooperativeBufferPipelineState: MTLComputePipelineState

    private let renderArgumentTable: any MTL4ArgumentTable
    private let renderCoopArgumentTable: any MTL4ArgumentTable
    private let cooperativeBufferArgumentTable: any MTL4ArgumentTable

    private let renderResidencySet: MTLResidencySet
    private let cooperativeBufferResidencySet: MTLResidencySet

    private let weights: InstantNGPMetalWeights
    private let renderUniformsBuffer: MTLBuffer
    private let numPositionsBuffer: MTLBuffer
    private let tensorArgumentsBuffer: MTLBuffer
    private let layerCountBuffer: MTLBuffer

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

        var mlpTensorArguments = MetalMLPTensorArguments()
        for (index, layer) in weights.mlp.layers.enumerated() {
            mlpTensorArguments.weight[index] = layer.weightTensor.gpuResourceID
            mlpTensorArguments.bias[index] = layer.biasTensor.gpuResourceID
        }

        guard let tensorArgumentsBuffer = device.makeBuffer(length: MemoryLayout<MetalMLPTensorArguments>.stride, options: .storageModeShared) else {
            throw InstantNGPError.failedToCreateBuffer
        }
        memcpy(tensorArgumentsBuffer.contents(), &mlpTensorArguments, MemoryLayout<MetalMLPTensorArguments>.stride)
        self.tensorArgumentsBuffer = tensorArgumentsBuffer

        var layerCount = UInt32(weights.mlp.layers.count)
        guard let layerCountBuffer = device.makeBuffer(length: MemoryLayout<UInt32>.stride, options: .storageModeShared) else {
            throw InstantNGPError.failedToCreateBuffer
        }
        memcpy(layerCountBuffer.contents(), &layerCount, MemoryLayout<UInt32>.stride)
        self.layerCountBuffer = layerCountBuffer

        let renderFn = MTL4LibraryFunctionDescriptor()
        renderFn.name = "instantNGPRender"
        renderFn.library = library
        let renderDesc = MTL4ComputePipelineDescriptor()
        renderDesc.computeFunctionDescriptor = renderFn
        self.renderPipelineState = try compiler.makeComputePipelineState(descriptor: renderDesc)

        let renderCoopFn = MTL4LibraryFunctionDescriptor()
        renderCoopFn.name = "instantNGPRenderCoop"
        renderCoopFn.library = library
        let renderCoopDesc = MTL4ComputePipelineDescriptor()
        renderCoopDesc.computeFunctionDescriptor = renderCoopFn
        self.renderCoopPipelineState = try compiler.makeComputePipelineState(descriptor: renderCoopDesc)

        let cooperativeBufferFn = MTL4LibraryFunctionDescriptor()
        cooperativeBufferFn.name = "instantNGPCoopBuffer"
        cooperativeBufferFn.library = library
        let cooperativeBufferDesc = MTL4ComputePipelineDescriptor()
        cooperativeBufferDesc.computeFunctionDescriptor = cooperativeBufferFn
        self.cooperativeBufferPipelineState = try compiler.makeComputePipelineState(descriptor: cooperativeBufferDesc)

        let renderTableDesc = MTL4ArgumentTableDescriptor()
        renderTableDesc.maxTextureBindCount = 1
        renderTableDesc.maxBufferBindCount = 4
        self.renderArgumentTable = try device.makeArgumentTable(descriptor: renderTableDesc)
        renderArgumentTable.setAddress(tensorArgumentsBuffer.gpuAddress, index: 0)
        renderArgumentTable.setAddress(layerCountBuffer.gpuAddress, index: 1)
        renderArgumentTable.setAddress(weights.hashTable.gpuAddress, index: 2)
        renderArgumentTable.setAddress(renderUniformsBuffer.gpuAddress, index: 3)

        let renderCoopTableDesc = MTL4ArgumentTableDescriptor()
        renderCoopTableDesc.maxTextureBindCount = 1
        renderCoopTableDesc.maxBufferBindCount = 5
        self.renderCoopArgumentTable = try device.makeArgumentTable(descriptor: renderCoopTableDesc)
        renderCoopArgumentTable.setAddress(tensorArgumentsBuffer.gpuAddress, index: 0)
        renderCoopArgumentTable.setAddress(layerCountBuffer.gpuAddress, index: 1)
        renderCoopArgumentTable.setAddress(weights.hashTable.gpuAddress, index: 2)
        renderCoopArgumentTable.setAddress(renderUniformsBuffer.gpuAddress, index: 3)
        renderCoopArgumentTable.setAddress(numPositionsBuffer.gpuAddress, index: 4)

        let cooperativeBufferTableDesc = MTL4ArgumentTableDescriptor()
        cooperativeBufferTableDesc.maxBufferBindCount = 6
        self.cooperativeBufferArgumentTable = try device.makeArgumentTable(descriptor: cooperativeBufferTableDesc)
        cooperativeBufferArgumentTable.setAddress(tensorArgumentsBuffer.gpuAddress, index: 0)
        cooperativeBufferArgumentTable.setAddress(layerCountBuffer.gpuAddress, index: 1)
        cooperativeBufferArgumentTable.setAddress(weights.hashTable.gpuAddress, index: 2)
        cooperativeBufferArgumentTable.setAddress(numPositionsBuffer.gpuAddress, index: 5)

        let renderResidency = try device.makeResidencySet(descriptor: .init())
        queue.addResidencySet(renderResidency)
        renderResidency.addAllocation(weights.hashTable)
        let addTensor: (MTLTensor, MTLResidencySet) -> Void = { tensor, residency in
            residency.addAllocation(tensor)
        }
        for layer in weights.mlp.layers {
            addTensor(layer.weightTensor, renderResidency)
            addTensor(layer.biasTensor, renderResidency)
        }
        renderResidency.addAllocation(renderUniformsBuffer)
        renderResidency.addAllocation(numPositionsBuffer)
        renderResidency.addAllocation(tensorArgumentsBuffer)
        renderResidency.addAllocation(layerCountBuffer)
        renderResidency.commit()
        self.renderResidencySet = renderResidency

        let cooperativeBufferResidency = try device.makeResidencySet(descriptor: .init())
        queue.addResidencySet(cooperativeBufferResidency)
        cooperativeBufferResidency.addAllocation(weights.hashTable)
        for layer in weights.mlp.layers {
            addTensor(layer.weightTensor, cooperativeBufferResidency)
            addTensor(layer.biasTensor, cooperativeBufferResidency)
        }
        cooperativeBufferResidency.addAllocation(numPositionsBuffer)
        cooperativeBufferResidency.addAllocation(tensorArgumentsBuffer)
        cooperativeBufferResidency.addAllocation(layerCountBuffer)
        cooperativeBufferResidency.commit()
        self.cooperativeBufferResidencySet = cooperativeBufferResidency
    }


    func supports(_ mode: RenderMode) -> Bool {
        switch mode {
        case .perPixel, .cooperative:
            return true
        }
    }

    func encode(drawableTexture: MTLTexture, commandBuffer: MTL4CommandBuffer, mode: RenderMode) {
        let pointer = renderUniformsBuffer.contents().bindMemory(to: RenderUniforms.self, capacity: 1)
        pointer.pointee.time = Float(CACurrentMediaTime())

        switch mode {
        case .perPixel:
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

        case .cooperative:
            encodeRenderCooperative(drawableTexture: drawableTexture, commandBuffer: commandBuffer)
        }
    }

    func encodeRenderCooperative(
        drawableTexture: MTLTexture,
        commandBuffer: MTL4CommandBuffer
    ) {
        let width = drawableTexture.width
        let height = drawableTexture.height
        guard width > 0 && height > 0 else { return }

        let pointer = renderUniformsBuffer.contents().bindMemory(to: RenderUniforms.self, capacity: 1)
        pointer.pointee.time = Float(CACurrentMediaTime())

        let numPixels = width * height
        var count = UInt32(numPixels)
        memcpy(numPositionsBuffer.contents(), &count, MemoryLayout<UInt32>.stride)

        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        commandBuffer.useResidencySet(renderResidencySet)
        encoder.setComputePipelineState(renderCoopPipelineState)
        encoder.setArgumentTable(renderCoopArgumentTable)
        renderCoopArgumentTable.setTexture(drawableTexture.gpuResourceID, index: 0)

        let threadsPerThreadgroup = MTLSize(
            width: renderCoopPipelineState.threadExecutionWidth * 4,
            height: 1,
            depth: 1
        )
        let numBatches = max(1, (numPixels + InstantNGPConfig.batchSize - 1) / InstantNGPConfig.batchSize)
        let threadgroupsPerGrid = MTLSize(width: numBatches, height: 1, depth: 1)

        encoder.dispatchThreadgroups(
            threadgroupsPerGrid: threadgroupsPerGrid,
            threadsPerThreadgroup: threadsPerThreadgroup
        )
        encoder.endEncoding()
    }

    func encodeCooperativeBuffer(
        positions: MTLTensor,
        outputs: MTLTensor,
        numPositions: UInt32,
        commandBuffer: MTL4CommandBuffer
    ) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        commandBuffer.useResidencySet(cooperativeBufferResidencySet)
        encoder.setComputePipelineState(cooperativeBufferPipelineState)
        encoder.setArgumentTable(cooperativeBufferArgumentTable)
        cooperativeBufferArgumentTable.setResource(positions.gpuResourceID, bufferIndex: 3)
        cooperativeBufferArgumentTable.setResource(outputs.gpuResourceID, bufferIndex: 4)
        var count = numPositions
        memcpy(numPositionsBuffer.contents(), &count, MemoryLayout<UInt32>.stride)

        let threadsPerThreadgroup = MTLSize(
            width: cooperativeBufferPipelineState.threadExecutionWidth * 4,
            height: 1,
            depth: 1
        )
        let numBatches = max(1, (Int(numPositions) + InstantNGPConfig.batchSize - 1) / InstantNGPConfig.batchSize)
        let threadgroupsPerGrid = MTLSize(width: 1, height: numBatches, depth: 1)
        encoder.dispatchThreadgroups(
            threadgroupsPerGrid: threadgroupsPerGrid,
            threadsPerThreadgroup: threadsPerThreadgroup
        )
        encoder.endEncoding()
    }
}
