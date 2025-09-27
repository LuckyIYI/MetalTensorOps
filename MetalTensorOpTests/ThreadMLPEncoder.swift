import Foundation
import Metal

struct ThreadMLPHostMetadata {
    var input: UInt32
    var hidden: UInt32
    var output: UInt32
}

final class ThreadMLPEncoder {
    private let pipelineState: MTLComputePipelineState
    private let argumentTable: any MTL4ArgumentTable
    private let residencySet: MTLResidencySet
    private let metadataBuffer: MTLBuffer

    init(
        device: MTLDevice,
        library: MTLLibrary,
        compiler: MTL4Compiler,
        queue: MTL4CommandQueue,
        weightsInputHidden: MTLTensor,
        weightsHiddenOutput: MTLTensor,
        input: MTLTensor,
        biasHidden: MTLTensor,
        biasOutput: MTLTensor,
        output: MTLTensor,
        inputDim: UInt32,
        hiddenDim: UInt32,
        outputDim: UInt32
    ) throws {
        let functionDescriptor = MTL4LibraryFunctionDescriptor()
        functionDescriptor.name = "threadMLP"
        functionDescriptor.library = library

        let pipelineDescriptor = MTL4ComputePipelineDescriptor()
        pipelineDescriptor.computeFunctionDescriptor = functionDescriptor
        self.pipelineState = try compiler.makeComputePipelineState(descriptor: pipelineDescriptor)

        let tableDescriptor = MTL4ArgumentTableDescriptor()
        tableDescriptor.maxBufferBindCount = 7
        self.argumentTable = try device.makeArgumentTable(descriptor: tableDescriptor)

        argumentTable.setResource(weightsInputHidden.gpuResourceID, bufferIndex: 0)
        argumentTable.setResource(weightsHiddenOutput.gpuResourceID, bufferIndex: 1)
        argumentTable.setResource(input.gpuResourceID, bufferIndex: 2)
        argumentTable.setResource(output.gpuResourceID, bufferIndex: 3)
        argumentTable.setResource(biasHidden.gpuResourceID, bufferIndex: 4)
        argumentTable.setResource(biasOutput.gpuResourceID, bufferIndex: 5)

        var metadata = ThreadMLPHostMetadata(input: inputDim, hidden: hiddenDim, output: outputDim)
        guard let metadataBuffer = device.makeBuffer(length: MemoryLayout<ThreadMLPHostMetadata>.stride, options: .storageModeShared) else {
            throw TestError("Failed to allocate metadata buffer")
        }
        memcpy(metadataBuffer.contents(), &metadata, MemoryLayout<ThreadMLPHostMetadata>.stride)
        self.metadataBuffer = metadataBuffer
        argumentTable.setAddress(metadataBuffer.gpuAddress, index: 6)

        let residency = try device.makeResidencySet(descriptor: .init())
        queue.addResidencySet(residency)

        func add(_ tensor: MTLTensor) {
            residency.addAllocation(tensor)
            if let buffer = tensor as? MTLBuffer {
                residency.addAllocation(buffer)
            }
        }

        add(weightsInputHidden)
        add(weightsHiddenOutput)
        add(input)
        add(output)
        add(biasHidden)
        add(biasOutput)
        residency.addAllocation(metadataBuffer)
        residency.commit()
        self.residencySet = residency
    }

    func encode(commandBuffer: MTL4CommandBuffer) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        commandBuffer.useResidencySet(residencySet)
        encoder.setComputePipelineState(pipelineState)
        encoder.setArgumentTable(argumentTable)
        let threads = MTLSize(width: 1, height: 1, depth: 1)
        encoder.dispatchThreads(threadsPerGrid: threads, threadsPerThreadgroup: threads)
        encoder.endEncoding()
    }
}
