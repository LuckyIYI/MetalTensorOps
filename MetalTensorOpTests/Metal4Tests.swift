import Foundation
import Testing
import Metal

struct TestError: Error, CustomStringConvertible {
    let description: String
    init(_ description: String) { self.description = description }
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

struct Metal4Tests {
    @Test func testSimdgroupMatrixMatrix() async throws {
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

        let encoder = try SimdMatrixMatrixEncoder(
            device: device,
            library: library,
            compiler: compiler,
            queue: commandQueue,
            tensorA: tensorA,
            tensorB: tensorB,
            tensorC: tensorC
        )

        commandBuffer.beginCommandBuffer(allocator: commandAllocator)

        let start = ContinuousClock.now

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

        let elapsed = start.duration(to: ContinuousClock.now)
        print("GPU time (SimdgroupMatmul): \(Double(elapsed.components.seconds) * 1000 + Double(elapsed.components.attoseconds) / 1e15) ms")

        var resultMatrix = [Float16](repeating: 0, count: M * N)
        let resultBufferPointer = bufferC.contents().bindMemory(to: Float16.self, capacity: M * N)
        resultMatrix = Array(UnsafeBufferPointer(start: resultBufferPointer, count: M * N))

//        print("Result matrix (first 10 elements): \(resultMatrix.prefix(10))")

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

    @Test func testThreadMatrixMatrix() async throws {
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
        let M = 64
        let N = 32
        let K = 32
        let sizeA = M * K * MemoryLayout<Float16>.size
        let sizeB = K * N * MemoryLayout<Float16>.size
        let sizeC = M * N * MemoryLayout<Float16>.size

        guard let bufferA = device.makeBuffer(length: sizeA, options: .storageModeShared) else { throw TestError("Could not create buffer A")}
        guard let bufferB = device.makeBuffer(length: sizeB, options: .storageModeShared) else { throw TestError("Could not create buffer B")}
        guard let bufferC = device.makeBuffer(length: sizeC, options: .storageModeShared) else { throw TestError("Could not create buffer C")}

        var matrixA = [Float16](repeating: 0, count: M * K)
        var matrixB = [Float16](repeating: 0, count: K * N)
        for i in 0..<(M * K) { matrixA[i] = Float16.random(in: 0...1) }
        for i in 0..<(K * N) { matrixB[i] = Float16.random(in: 0...1) }

        bufferA.contents().copyMemory(from: matrixA, byteCount: sizeA)
        bufferB.contents().copyMemory(from: matrixB, byteCount: sizeB)
        bufferC.contents().initializeMemory(as: UInt8.self, repeating: 0, count: sizeC)

        let tensorA = try makeTensor(from: bufferA, rows: M, columns: K)
        let tensorB = try makeTensor(from: bufferB, rows: K, columns: N)
        let tensorC = try makeTensor(from: bufferC, rows: M, columns: N)
        guard let commandBuffer = device.makeCommandBuffer() else {
            throw TestError("Could not create command buffer")
        }
        let encoder = try ThreadMatrixMatrixEncoder(
            device: device,
            library: library,
            compiler: compiler,
            queue: commandQueue,
            tensorA: tensorA,
            tensorB: tensorB,
            tensorC: tensorC
        )
        commandBuffer.beginCommandBuffer(allocator: commandAllocator)
        let start = ContinuousClock.now
        encoder.encode(commandBuffer: commandBuffer)
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
        let elapsed = start.duration(to: ContinuousClock.now)
        print("GPU time (ThreadMatmul): \(Double(elapsed.components.seconds) * 1000 + Double(elapsed.components.attoseconds) / 1e15) ms")
        let resultBufferPointer = bufferC.contents().bindMemory(to: Float16.self, capacity: M * N)
        let resultMatrix = Array(UnsafeBufferPointer(start: resultBufferPointer, count: M * N))

        var cpuC = [Float](repeating: 0, count: M * N)
        for m in 0..<M {
            for n in 0..<N {
                var sum: Float = 0
                for k in 0..<K {
                    let a = Float(matrixA[m * K + k])
                    let b = Float(matrixB[k * N + n])
                    sum += a * b
                }
                cpuC[m * N + n] = sum
            }
        }

        let epsilon: Float = 1e-1
        for (row, col) in [(0,0), (M-1,N-1), (M/2,N/2)] {
            let gpuValue = Float(resultMatrix[row * N + col])
            let cpuValue = cpuC[row * N + col]
            print("C[\(row),\(col)]: GPU=\(gpuValue), CPU=\(cpuValue)")
            #expect(abs(gpuValue - cpuValue) < epsilon, "Mismatch at (\(row),\(col))")
        }
    }

    @Test func testThreadVectorMatix() async throws {
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
        // Test: vector (1 x K) × matrix (K x N) = (1 x N)
        let K = 31, N = 17
        let sizeA = 1 * K * MemoryLayout<Float16>.size
        let sizeB = K * N * MemoryLayout<Float16>.size
        let sizeC = 1 * N * MemoryLayout<Float16>.size
        guard let bufferA = device.makeBuffer(length: sizeA, options: .storageModeShared) else { throw TestError("Could not create buffer A")}
        guard let bufferB = device.makeBuffer(length: sizeB, options: .storageModeShared) else { throw TestError("Could not create buffer B")}
        guard let bufferC = device.makeBuffer(length: sizeC, options: .storageModeShared) else { throw TestError("Could not create buffer C")}

        var matrixA = [Float16](repeating: 0, count: K)
        var matrixB = [Float16](repeating: 0, count: K*N)
        for i in 0..<K { matrixA[i] = Float16.random(in: 0...1) }
        for i in 0..<(K*N) { matrixB[i] = Float16.random(in: 0...1) }

        bufferA.contents().copyMemory(from: matrixA, byteCount: sizeA)
        bufferB.contents().copyMemory(from: matrixB, byteCount: sizeB)
        bufferC.contents().initializeMemory(as: UInt8.self, repeating: 0, count: sizeC)

        let tensorA = try makeTensor(from: bufferA, rows: 1, columns: K)
        let tensorB = try makeTensor(from: bufferB, rows: K, columns: N)
        let tensorC = try makeTensor(from: bufferC, rows: 1, columns: N)

        guard let commandBuffer = device.makeCommandBuffer() else {
            throw TestError("Could not create command buffer")
        }

        let encoder = try ThreadVectorMatrixEncoder(
            device: device,
            library: library,
            compiler: compiler,
            queue: commandQueue,
            tensorA: tensorA,
            tensorB: tensorB,
            tensorC: tensorC
        )

        commandBuffer.beginCommandBuffer(allocator: commandAllocator)
        let start = ContinuousClock.now
        encoder.encode(commandBuffer: commandBuffer)
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
    
        let elapsed = start.duration(to: ContinuousClock.now)
        print("GPU time (VectorMatmul): \(Double(elapsed.components.seconds) * 1000 + Double(elapsed.components.attoseconds) / 1e15) ms")
        let resultBufferPointer = bufferC.contents().bindMemory(to: Float16.self, capacity: N)
        let resultVector = Array(UnsafeBufferPointer(start: resultBufferPointer, count: N))

        var cpuC = [Float](repeating: 0, count: N)
        for n in 0..<N {
            var sum: Float = 0
            for k in 0..<K {
                let a = Float(matrixA[k])
                let b = Float(matrixB[k*N + n])
                sum += a * b
            }
            cpuC[n] = sum
        }

        let epsilon: Float = 1e-1
        for n in [0, N/2, N-1] {
            let gpuValue = Float(resultVector[n])
            let cpuValue = cpuC[n]
            print("C[0,\(n)]: GPU=\(gpuValue), CPU=\(cpuValue)")
            #expect(abs(gpuValue - cpuValue) < epsilon, "Mismatch at (0,\(n))")
        }
    }
    
    @Test func testThreadMatrixVector() async throws {
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
        // Test: matrix (M x K) × vector (K x 1) = vector (M x 1)
        let M = 29, K = 19
        let sizeA = M * K * MemoryLayout<Float16>.size
        let sizeB = K * 1 * MemoryLayout<Float16>.size
        let sizeC = M * 1 * MemoryLayout<Float16>.size
        guard let bufferA = device.makeBuffer(length: sizeA, options: .storageModeShared) else { throw TestError("Could not create buffer A")}
        guard let bufferB = device.makeBuffer(length: sizeB, options: .storageModeShared) else { throw TestError("Could not create buffer B")}
        guard let bufferC = device.makeBuffer(length: sizeC, options: .storageModeShared) else { throw TestError("Could not create buffer C")}

        var matrixA = [Float16](repeating: 0, count: M * K)
        var matrixB = [Float16](repeating: 0, count: K)
        for i in 0..<(M * K) { matrixA[i] = Float16.random(in: 0...1) }
        for i in 0..<K { matrixB[i] = Float16.random(in: 0...1) }

        bufferA.contents().copyMemory(from: matrixA, byteCount: sizeA)
        bufferB.contents().copyMemory(from: matrixB, byteCount: sizeB)
        bufferC.contents().initializeMemory(as: UInt8.self, repeating: 0, count: sizeC)

        let tensorA = try makeTensor(from: bufferA, rows: M, columns: K)
        let tensorB = try makeTensor(from: bufferB, rows: K, columns: 1)
        let tensorC = try makeTensor(from: bufferC, rows: M, columns: 1)

        guard let commandBuffer = device.makeCommandBuffer() else {
            throw TestError("Could not create command buffer")
        }

        let encoder = try ThreadMatrixVectorEncoder(
            device: device,
            library: library,
            compiler: compiler,
            queue: commandQueue,
            tensorA: tensorA,
            tensorB: tensorB,
            tensorC: tensorC
        )

        commandBuffer.beginCommandBuffer(allocator: commandAllocator)
        let start = ContinuousClock.now
        encoder.encode(commandBuffer: commandBuffer)
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
    
        let elapsed = start.duration(to: ContinuousClock.now)
        print("GPU time (ThreadMatrixVector): \(Double(elapsed.components.seconds) * 1000 + Double(elapsed.components.attoseconds) / 1e15) ms")
        let resultBufferPointer = bufferC.contents().bindMemory(to: Float16.self, capacity: M)
        let resultVector = Array(UnsafeBufferPointer(start: resultBufferPointer, count: M))

        var cpuC = [Float](repeating: 0, count: M)
        for m in 0..<M {
            var sum: Float = 0
            for k in 0..<K {
                let a = Float(matrixA[m * K + k])
                let b = Float(matrixB[k])
                sum += a * b
            }
            cpuC[m] = sum
        }

        let epsilon: Float = 1e-1
        for m in [0, M/2, M-1] {
            let gpuValue = Float(resultVector[m])
            let cpuValue = cpuC[m]
            print("C[\(m),0]: GPU=\(gpuValue), CPU=\(cpuValue)")
            #expect(abs(gpuValue - cpuValue) < epsilon, "Mismatch at (\(m),0)")
        }
    }
}

