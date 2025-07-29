import Testing
import Metal

struct TestError: Error, CustomStringConvertible {
    let description: String
    init(_ description: String) { self.description = description }
}

class MatMulEncoder {
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
        functionDescriptor.name = "simdgroupMatmul"
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

        let threadgroups = MTLSize(width: (N + 31) / 32, height: (M + 63) / 64, depth: 1)
        let simdgroupWidth = pipelineState.threadExecutionWidth
        let threadsPerThreadgroup = MTLSize(width: simdgroupWidth * 4, height: 1, depth: 1)

        encoder.dispatchThreadgroups(threadgroupsPerGrid: threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
        encoder.endEncoding()
    }
}

func makeTensor(from buffer: MTLBuffer, rows: Int, columns: Int) throws -> MTLTensor {
    guard let extents = MTLTensorExtents([columns, rows]) else {
        throw TestError("Failed to create extents")
    }

    guard let strides = MTLTensorExtents([1, columns]) else {
        throw TestError("Failed to create strides")
    }
    let descriptor = MTLTensorDescriptor()
    descriptor.dimensions = extents
    descriptor.usage = .compute
    descriptor.strides  = strides
    descriptor.dataType = .float16
    return try buffer.makeTensor(descriptor: descriptor, offset: 0)
}

struct MetalTensorOpTests {
    @Test func testCooperativeMatmul() async throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw TestError("Metal is not supported on this device")
        }

        guard device.supportsFamily(.metal4) else {
            throw TestError("This device does not support the tensor operations used in the shader.")
        }


        guard let commandQueue = device.makeMTL4CommandQueue() else {
            throw TestError("Could not create command queue")
        }
        
        guard let commandAllocator = device.makeCommandAllocator() else {
            throw TestError("Could not create command allocator")
        }
        
        let compiler = try device.makeCompiler(descriptor: .init())
        let library = try device.makeDefaultLibrary(bundle: .main)

        let M = 128
        let N = 64
        let K = 256

        let sizeA = M * K * MemoryLayout<Float16>.size
        let sizeB = K * N * MemoryLayout<Float16>.size
        let sizeC = M * N * MemoryLayout<Float16>.size

        guard let bufferA = device.makeBuffer(length: sizeA, options: .storageModeShared) else { throw TestError("Could not create buffer A")}
        guard let bufferB = device.makeBuffer(length: sizeB, options: .storageModeShared) else { throw TestError("Could not create buffer B")}
        guard let bufferC = device.makeBuffer(length: sizeC, options: .storageModeShared) else { throw TestError("Could not create buffer C") }

        var matrixA = [Float16](repeating: 0, count: M * K)
        var matrixB = [Float16](repeating: 0, count: K * N)
        for i in 0..<(M * K) {
            matrixA[i] = Float16.random(in: 0...1)
        }
        for i in 0..<(K * N) {
            matrixB[i] = Float16.random(in: 0...1)
        }

        bufferA.contents().copyMemory(from: matrixA, byteCount: sizeA)
        bufferB.contents().copyMemory(from: matrixB, byteCount: sizeB)
        bufferC.contents().initializeMemory(as: UInt8.self, repeating: 0, count: sizeC)

        let rowsA = M; let colsA = K
        let rowsB = K; let colsB = N
        let rowsC = M; let colsC = N
        

        let tensorA = try makeTensor(from: bufferA, rows: rowsA, columns: colsA)
        let tensorB = try makeTensor(from: bufferB, rows: rowsB, columns: colsB)
        let tensorC = try makeTensor(from: bufferC, rows: rowsC, columns: colsC)

        guard let commandBuffer = device.makeCommandBuffer() else {
            throw TestError("Could not create command buffer")
        }
        
        let encoder = try MatMulEncoder(
            device: device,
            library: library,
            compiler: compiler,
            queue: commandQueue,
            tensorA: tensorA,
            tensorB: tensorB,
            tensorC: tensorC
        )
        
        commandBuffer.beginCommandBuffer(allocator: commandAllocator)
        encoder.encode(commandBuffer: commandBuffer, M: M, N: N)
        commandBuffer.endCommandBuffer()

        let commitOptionsCol = MTL4CommitOptions()

        try await withCheckedThrowingContinuation { (cont: CheckedContinuation<Void, Error>) in
            commitOptionsCol.addFeedbackHandler { feedback in
                if feedback.error == nil {
                    cont.resume()
                } else {
                    cont.resume(throwing: TestError("GPU execution failed with error: \(feedback.error!)"))
                }
            }
            commandQueue.commit([commandBuffer], options: commitOptionsCol)
        }
        
        var resultMatrix = [Float16](repeating: 0, count: M * N)
        let resultBufferPointer = bufferC.contents().bindMemory(to: Float16.self, capacity: M * N)
        resultMatrix = Array(UnsafeBufferPointer(start: resultBufferPointer, count: M * N))

        print("Result matrix (first 10 elements): \(resultMatrix.prefix(10))")

        let epsilon: Float = 1e-1
        let testIndices = [(0,0), (0,1), (1,0), (M-1, N-1), (M/2, N/2)]
        for (row, col) in testIndices {
            var expected: Float = 0
            for k in 0..<K {
                let a = Float(matrixA[row * K + k])
                let b = Float(matrixB[k * N + col])
                expected += a * b
            }
            let gpuValue = Float(resultMatrix[row * N + col])
            print("C[\(row),\(col)]: expected = \(expected), real = \(gpuValue)")
            #expect(abs(gpuValue - expected) < epsilon, "C[\(row),\(col)] mismatch: CPU = \(expected), GPU = \(gpuValue)")
        }
    }
}
