import Foundation
import Metal

class ThreadVectorMatrixEncoder {
    let pipelineState: MTLComputePipelineState
    var argumentTable: any MTL4ArgumentTable
    let device: MTLDevice
    let residencySet: MTLResidencySet

    init(
        device: MTLDevice,
        library: MTLLibrary,
        compiler: MTL4Compiler,
        queue: MTL4CommandQueue,
        tensorA: MTLTensor, // (1 x K)
        tensorB: MTLTensor, // (K x N)
        tensorC: MTLTensor  // (1 x N)
    ) throws {
        self.device = device
        let functionDescriptor = MTL4LibraryFunctionDescriptor()
        functionDescriptor.name = "threadVectorMatrix"
        functionDescriptor.library = library
        let pipelineDescriptor = MTL4ComputePipelineDescriptor()
        pipelineDescriptor.computeFunctionDescriptor = functionDescriptor
        self.pipelineState = try compiler.makeComputePipelineState(descriptor: pipelineDescriptor)

        let tableDesc = MTL4ArgumentTableDescriptor()
        tableDesc.maxBufferBindCount = 3

        self.argumentTable = try device.makeArgumentTable(descriptor: tableDesc)
        self.argumentTable.setResource(tensorA.gpuResourceID, bufferIndex: 0)
        self.argumentTable.setResource(tensorB.gpuResourceID, bufferIndex: 1)
        self.argumentTable.setResource(tensorC.gpuResourceID, bufferIndex: 2)
        
        let residency = try device.makeResidencySet(descriptor: .init())
        queue.addResidencySet(residency)

        residency.addAllocation(tensorA)
        residency.addAllocation(tensorB)
        residency.addAllocation(tensorC)
        residency.commit()

        self.residencySet = residency
    }

    func encode(commandBuffer: MTL4CommandBuffer) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        commandBuffer.useResidencySet(residencySet)
        encoder.setComputePipelineState(pipelineState)
        encoder.setArgumentTable(argumentTable)

        let threadsPerGrid = MTLSize(width: 1, height: 1, depth: 1)
        let threadsPerThreadgroup = MTLSize(width: 1, height: 1, depth: 1)
        encoder.dispatchThreads(threadsPerGrid: threadsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
        encoder.endEncoding()
    }
}
