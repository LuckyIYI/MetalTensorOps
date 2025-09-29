import Foundation
import Metal

struct DynamicMLPTensorArguments {
    var weight: InlineArray<8, MTLResourceID>
    var bias: InlineArray<8, MTLResourceID>

    init() {
        weight = .init(repeating: .init())
        bias = .init(repeating: .init())
    }
}

struct ThreadDynamicMLPHostMetadata {
    var inputDim: UInt32
    var hiddenDim: UInt32
    var outputDim: UInt32
    var hiddenLayerCount: UInt32 // Number of hidden layers
}

final class ThreadDynamicMLPEncoder {
    private let pipelineState: MTLComputePipelineState
    private let argumentTable: any MTL4ArgumentTable
    private let residencySet: MTLResidencySet
    private let metadataBuffer: MTLBuffer
    private let layersArgumentsBuffer: MTLBuffer

    init(
        device: MTLDevice,
        library: MTLLibrary,
        compiler: MTL4Compiler,
        queue: MTL4CommandQueue,
        weightTensors: [MTLTensor],
        biasTensors: [MTLTensor],
        input: MTLTensor,
        output: MTLTensor,
        inputDim: UInt32,
        hiddenDim: UInt32,
        outputDim: UInt32,
        hiddenLayerCount: UInt32
    ) throws {
        let functionDescriptor = MTL4LibraryFunctionDescriptor()
        functionDescriptor.name = "threadDynamicMLP"
        functionDescriptor.library = library

        let pipelineDescriptor = MTL4ComputePipelineDescriptor()
        pipelineDescriptor.computeFunctionDescriptor = functionDescriptor
        self.pipelineState = try compiler.makeComputePipelineState(descriptor: pipelineDescriptor)

        // Create layers struct with tensor resource IDs (like SirenEncoder does)
        var layersArguments = DynamicMLPTensorArguments()
        for (i, wt) in weightTensors.enumerated() where i < 8 {
            layersArguments.weight[i] = wt.gpuResourceID
        }
        for (i, bt) in biasTensors.enumerated() where i < 8 {
            layersArguments.bias[i] = bt.gpuResourceID
        }

        guard let layersArgumentsBuffer = device.makeBuffer(length: MemoryLayout<DynamicMLPTensorArguments>.stride, options: .storageModeShared) else {
            throw TestError("Failed to allocate layers arguments buffer")
        }
        self.layersArgumentsBuffer = layersArgumentsBuffer
        memcpy(layersArgumentsBuffer.contents(), &layersArguments, MemoryLayout<DynamicMLPTensorArguments>.stride)

        // Create metadata
        var metadata = ThreadDynamicMLPHostMetadata(
            inputDim: inputDim,
            hiddenDim: hiddenDim,
            outputDim: outputDim,
            hiddenLayerCount: hiddenLayerCount
        )
        guard let metadataBuffer = device.makeBuffer(length: MemoryLayout<ThreadDynamicMLPHostMetadata>.stride, options: .storageModeShared) else {
            throw TestError("Failed to allocate metadata buffer")
        }
        memcpy(metadataBuffer.contents(), &metadata, MemoryLayout<ThreadDynamicMLPHostMetadata>.stride)
        self.metadataBuffer = metadataBuffer

        // Setup argument table
        let tableDescriptor = MTL4ArgumentTableDescriptor()
        tableDescriptor.maxBufferBindCount = 4
        self.argumentTable = try device.makeArgumentTable(descriptor: tableDescriptor)

        argumentTable.setAddress(layersArgumentsBuffer.gpuAddress, index: 0)
        argumentTable.setResource(input.gpuResourceID, bufferIndex: 1)
        argumentTable.setResource(output.gpuResourceID, bufferIndex: 2)
        argumentTable.setAddress(metadataBuffer.gpuAddress, index: 3)

        // Add all resources to residency
        let residency = try device.makeResidencySet(descriptor: .init())
        queue.addResidencySet(residency)

        func add(_ tensor: MTLTensor) {
            residency.addAllocation(tensor)
            if let buffer = tensor as? MTLBuffer {
                residency.addAllocation(buffer)
            }
        }

        for wt in weightTensors {
            add(wt)
        }
        for bt in biasTensors {
            add(bt)
        }
        add(input)
        add(output)
        residency.addAllocation(metadataBuffer)
        residency.addAllocation(layersArgumentsBuffer)
        residency.commit()
        self.residencySet = residency
    }

    func encode(commandBuffer: MTL4CommandBuffer, threadCount: Int = 1) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        commandBuffer.useResidencySet(residencySet)
        encoder.setComputePipelineState(pipelineState)
        encoder.setArgumentTable(argumentTable)
        let threads = MTLSize(width: threadCount, height: 1, depth: 1)
        let threadgroup = MTLSize(width: min(threadCount, 32), height: 1, depth: 1)
        encoder.dispatchThreads(threadsPerGrid: threads, threadsPerThreadgroup: threadgroup)
        encoder.endEncoding()
    }
}