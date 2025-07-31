import Foundation
import Metal

class SimdMatrixMatrixEncoder {
    let pipelineState: MTLComputePipelineState
    var argumentTable: any MTL4ArgumentTable
    let device: MTLDevice
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
        self.device = device
        let functionDescriptor = MTL4LibraryFunctionDescriptor()
        functionDescriptor.name = "simdgroupMatrixMatrix"
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

    func encode(commandBuffer: MTL4CommandBuffer, M: Int, N: Int) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        commandBuffer.useResidencySet(residencySet)
        encoder.setComputePipelineState(pipelineState)
        encoder.setArgumentTable(argumentTable)

        // Each compute tile covers 32 columns × 64 rows of the output matrix.
        let tileWidth  = 32
        let tileHeight = 64

        // Number of tiles required to cover the full output.
        let tilesPerGrid = MTLSize(
            width:  (N + tileWidth  - 1) / tileWidth,
            height: (M + tileHeight - 1) / tileHeight,
            depth:  1
        )

        let simdgroupWidth = pipelineState.threadExecutionWidth
        let threadsPerGroup = MTLSize(width: simdgroupWidth * 4, height: 1, depth:  1)

        encoder.dispatchThreadgroups(threadgroupsPerGrid: tilesPerGrid, threadsPerThreadgroup: threadsPerGroup)
        encoder.endEncoding()
    }
}
