import Foundation
import Metal

class SimdMatrixMatrixMetal3Encoder {
    let pipelineState: MTLComputePipelineState
    let device: MTLDevice
    let functionName = "simdgroupMatrixMatrixMetal3"
    let bufferA: MTLBuffer
    let bufferB: MTLBuffer
    let bufferC: MTLBuffer

    init(device: MTLDevice, library: MTLLibrary, bufferA: MTLBuffer, bufferB: MTLBuffer, bufferC: MTLBuffer) throws {
        self.device = device
        self.bufferA = bufferA
        self.bufferB = bufferB
        self.bufferC = bufferC

        guard let function = library.makeFunction(name: functionName) else {
            throw TestError("Kernel function '\(functionName)' not found")
        }
        self.pipelineState = try device.makeComputePipelineState(function: function)
    }

    func encode(commandBuffer: MTLCommandBuffer, M: Int, N: Int, K: Int) throws {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw TestError("Failed to create command encoder")
        }

        encoder.setComputePipelineState(pipelineState)
        encoder.setBuffer(bufferA, offset: 0, index: 0)
        encoder.setBuffer(bufferB, offset: 0, index: 1)
        encoder.setBuffer(bufferC, offset: 0, index: 2)

        // One 8×8 output tile per SIMD-group → one thread-group (32 threads) per tile.
        let threadsPerThreadgroup = MTLSize(width: pipelineState.threadExecutionWidth, height: 1, depth: 1)

        // Each tile covers 8 columns × 8 rows of output matrix C
        let tilesX = N / 8
        let tilesY = M / 8

        let threadgroupsPerGrid = MTLSize(width: tilesX, height: tilesY, depth: 1)

        encoder.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
        encoder.endEncoding()
    }
}
