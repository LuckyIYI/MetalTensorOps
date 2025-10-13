import Foundation
import Metal

class SimdMatrixVectorEncoder {
    let pipelineState: MTLComputePipelineState
    var argumentTable: any MTL4ArgumentTable
    let residencySet: MTLResidencySet

    init(
        device: MTLDevice,
        library: MTLLibrary,
        compiler: MTL4Compiler,
        queue: MTL4CommandQueue,
        tensorA: MTLTensor,
        tensorB: MTLTensor,
        tensorC: MTLTensor
    ) throws {
        var functionDescriptor = MTL4LibraryFunctionDescriptor()
        functionDescriptor.name = "simdgroupMatrixVector"
        functionDescriptor.library = library

        let pipelineDescriptor = MTL4ComputePipelineDescriptor()
        pipelineDescriptor.computeFunctionDescriptor = functionDescriptor
        self.pipelineState = try compiler.makeComputePipelineState(descriptor: pipelineDescriptor)

        let tableDescriptor = MTL4ArgumentTableDescriptor()
        tableDescriptor.maxBufferBindCount = 3
        self.argumentTable = try device.makeArgumentTable(descriptor: tableDescriptor)

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

    func encode(commandBuffer: MTL4CommandBuffer, rows: Int) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        commandBuffer.useResidencySet(residencySet)
        encoder.setComputePipelineState(pipelineState)
        encoder.setArgumentTable(argumentTable)

        let tileHeight = 64
        let groups = (rows + tileHeight - 1) / tileHeight

        let simdWidth = pipelineState.threadExecutionWidth
        let threadsPerThreadgroup = MTLSize(width: simdWidth * 4, height: 1, depth: 1)
        let threadgroupsPerGrid = MTLSize(width: groups, height: 1, depth: 1)

        encoder.dispatchThreadgroups(threadgroupsPerGrid: threadgroupsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
        encoder.endEncoding()
    }
}
