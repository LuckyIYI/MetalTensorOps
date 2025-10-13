import Foundation
import Metal

private struct GemvParameters {
    var rows: UInt32
    var cols: UInt32
    var rowsPerSimdgroup: UInt32
    var simdgroupsPerThreadgroup: UInt32
}

class SimdHierarchicalMatrixVectorEncoder {
    let pipelineState: MTLComputePipelineState
    var argumentTable: any MTL4ArgumentTable
    let residencySet: MTLResidencySet
    let configBuffer: MTLBuffer
    private let maxRowsPerSimdGroup = 8
    private let maxSimdGroupsPerThreadgroup = 8

    private let bufferA: MTLBuffer
    private let bufferB: MTLBuffer
    private let bufferC: MTLBuffer

    init(
        device: MTLDevice,
        library: MTLLibrary,
        compiler: MTL4Compiler,
        queue: MTL4CommandQueue,
        bufferA: MTLBuffer,
        bufferB: MTLBuffer,
        bufferC: MTLBuffer
    ) throws {
        self.bufferA = bufferA
        self.bufferB = bufferB
        self.bufferC = bufferC

        let functionDescriptor = MTL4LibraryFunctionDescriptor()
        functionDescriptor.name = "simdHierarchicalMatrixVector"
        functionDescriptor.library = library

        let pipelineDescriptor = MTL4ComputePipelineDescriptor()
        pipelineDescriptor.computeFunctionDescriptor = functionDescriptor
        self.pipelineState = try compiler.makeComputePipelineState(descriptor: pipelineDescriptor)

        guard let configBuffer = device.makeBuffer(length: MemoryLayout<GemvParameters>.stride, options: .storageModeShared) else {
            throw TestError("Failed to allocate GEMV config buffer")
        }
        self.configBuffer = configBuffer

        let tableDescriptor = MTL4ArgumentTableDescriptor()
        tableDescriptor.maxBufferBindCount = 4
        self.argumentTable = try device.makeArgumentTable(descriptor: tableDescriptor)

        self.argumentTable.setAddress(bufferA.gpuAddress, index: 0)
        self.argumentTable.setAddress(bufferB.gpuAddress, index: 1)
        self.argumentTable.setAddress(bufferC.gpuAddress, index: 2)
        self.argumentTable.setAddress(configBuffer.gpuAddress, index: 3)

        let residency = try device.makeResidencySet(descriptor: .init())
        queue.addResidencySet(residency)
        residency.addAllocation(bufferA)
        residency.addAllocation(bufferB)
        residency.addAllocation(bufferC)
        residency.addAllocation(configBuffer)
        residency.commit()
        self.residencySet = residency
    }

    func encode(commandBuffer: MTL4CommandBuffer, rows: Int, columns: Int) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }

        let tuning = tunedKernelShape(rows: rows, columns: columns)

        var parameters = GemvParameters(
            rows: UInt32(rows),
            cols: UInt32(columns),
            rowsPerSimdgroup: UInt32(tuning.rowsPerSimdgroup),
            simdgroupsPerThreadgroup: UInt32(tuning.simdgroupsPerThreadgroup)
        )
        memcpy(configBuffer.contents(), &parameters, MemoryLayout<GemvParameters>.stride)

        commandBuffer.useResidencySet(residencySet)
        encoder.setComputePipelineState(pipelineState)
        encoder.setArgumentTable(argumentTable)

        let simdWidth = pipelineState.threadExecutionWidth
        let threadsPerThreadgroup = MTLSize(width: simdWidth * tuning.simdgroupsPerThreadgroup, height: 1, depth: 1)
        let rowsPerThreadgroup = tuning.rowsPerSimdgroup * tuning.simdgroupsPerThreadgroup
        let groups = (rows + rowsPerThreadgroup - 1) / rowsPerThreadgroup
        let threadgroupsPerGrid = MTLSize(width: groups, height: 1, depth: 1)
        encoder.dispatchThreadgroups(threadgroupsPerGrid: threadgroupsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
        encoder.endEncoding()
    }

    private func tunedKernelShape(rows: Int, columns: Int) -> (rowsPerSimdgroup: Int, simdgroupsPerThreadgroup: Int) {
        let rowsPerSimdgroup: Int = {
            if rows >= 1024 {
                return 4
            } else if rows >= 512 {
                return 3
            } else if rows >= 256 {
                return 2
            } else {
                return 1
            }
        }()

        let simdgroupsPerThreadgroup: Int = {
            if columns >= 4096 {
                return 8
            } else if columns >= 2048 {
                return 6
            } else if columns >= 1024 {
                return 4
            } else {
                return 2
            }
        }()

        let clampedRowsPerSimdgroup = min(maxRowsPerSimdGroup, max(1, rowsPerSimdgroup))
        let maxGroupsNeeded = max(1, (rows + clampedRowsPerSimdgroup - 1) / clampedRowsPerSimdgroup)
        let desiredGroups = min(simdgroupsPerThreadgroup, maxGroupsNeeded)
        let clampedGroups = min(maxSimdGroupsPerThreadgroup, max(1, desiredGroups))

        return (clampedRowsPerSimdgroup, clampedGroups)
    }
}
