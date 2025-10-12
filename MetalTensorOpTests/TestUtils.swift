import Foundation
import Metal
import Testing

struct TestError: Error, CustomStringConvertible {
    let description: String
    init(_ description: String) { self.description = description }
}

enum TensorMemoryLayout {
    case columnMajor
    case rowMajor
}

private let repositoryRootURL: URL = {
    var url = URL(fileURLWithPath: #filePath)
    while url.lastPathComponent != "MetalTensorOpTests" && url.pathComponents.count > 1 {
        url.deleteLastPathComponent()
    }
    if url.lastPathComponent == "MetalTensorOpTests" {
        url.deleteLastPathComponent()
    }
    return url
}()

private let sharedDataFolderURL: URL = {
    repositoryRootURL.appendingPathComponent("MetalTensorOp/Data", isDirectory: true)
}()

func locateDataResource(named resource: String, withExtension ext: String = "json") throws -> URL {
    if let bundle = Bundle.allBundles.first(where: { $0.bundlePath.contains("MetalTensorOpTests.xctest") }) {
        if let url = bundle.url(forResource: resource, withExtension: ext) {
            return url
        }
    }

    let candidate = sharedDataFolderURL.appendingPathComponent(resource).appendingPathExtension(ext)
    if FileManager.default.fileExists(atPath: candidate.path) {
        return candidate
    }

    throw TestError("Resource \(resource).\(ext) not found in test bundle or shared data folder at \(candidate.path)")
}

func makeTensor(
    from buffer: MTLBuffer,
    rows: Int,
    columns: Int,
    layout: TensorMemoryLayout = .columnMajor,
    dataType: MTLTensorDataType = .float16
) throws -> MTLTensor {
    let extents: MTLTensorExtents?
    let strides: MTLTensorExtents?

    switch layout {
    case .columnMajor:
        extents = MTLTensorExtents([columns, rows])
        strides = MTLTensorExtents([1, columns])
    case .rowMajor:
        extents = MTLTensorExtents([rows, columns])
        strides = MTLTensorExtents([1, rows])
    }

    guard let tensorExtents = extents else {
        throw TestError("Failed to create extents for a \(rows)x\(columns) tensor")
    }

    guard let tensorStrides = strides else {
        throw TestError("Failed to create strides for a \(rows)x\(columns) tensor")
    }

    let descriptor = MTLTensorDescriptor()
    descriptor.dimensions = tensorExtents
    descriptor.usage = .compute
    descriptor.strides = tensorStrides
    descriptor.dataType = dataType
    return try buffer.makeTensor(descriptor: descriptor, offset: 0)
}

func makeVectorTensor(from buffer: MTLBuffer, length: Int) throws -> MTLTensor {
    guard let extents = MTLTensorExtents([length]) else {
        throw TestError("Failed to create extents for a vector of length \(length)")
    }
    guard let strides = MTLTensorExtents([1]) else {
        throw TestError("Failed to create strides for a vector of length \(length)")
    }

    let descriptor = MTLTensorDescriptor()
    descriptor.dimensions = extents
    descriptor.usage = .compute
    descriptor.strides = strides
    descriptor.dataType = .float16
    return try buffer.makeTensor(descriptor: descriptor, offset: 0)
}

struct TestContext {
    let device: MTLDevice
    let commandQueue: MTL4CommandQueue
    let commandAllocator: MTL4CommandAllocator
    let compiler: MTL4Compiler
    let library: MTLLibrary

    init() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw TestError("Metal is not supported on this device")
        }
        self.device = device

        guard device.supportsFamily(.metal4) else {
            throw TestError("This device does not support the tensor operations used in the shader.")
        }

        guard let commandQueue = device.makeMTL4CommandQueue() else {
            throw TestError("Could not create command queue")
        }
        self.commandQueue = commandQueue

        guard let commandAllocator = device.makeCommandAllocator() else {
            throw TestError("Could not create command allocator")
        }
        self.commandAllocator = commandAllocator

        self.compiler = try device.makeCompiler(descriptor: .init())
        self.library = try device.makeDefaultLibrary(bundle: .main)
    }
}

func setupMatrixMatrixTest(device: MTLDevice, M: Int, N: Int, K: Int) throws -> (bufferA: MTLBuffer, bufferB: MTLBuffer, bufferC: MTLBuffer, matrixA: [Float16], matrixB: [Float16]) {
    let sizeA = M * K * MemoryLayout<Float16>.size
    let sizeB = K * N * MemoryLayout<Float16>.size
    let sizeC = M * N * MemoryLayout<Float16>.size

    guard let bufferA = device.makeBuffer(length: sizeA, options: .storageModeShared),
          let bufferB = device.makeBuffer(length: sizeB, options: .storageModeShared),
          let bufferC = device.makeBuffer(length: sizeC, options: .storageModeShared) else {
        throw TestError("Could not create buffers for Matrix-Matrix test")
    }

    let matrixA = [Float16](unsafeUninitializedCapacity: M * K) { buffer, initializedCount in
        for i in 0..<(M * K) { buffer[i] = Float16.random(in: 0...1) }
        initializedCount = M * K
    }
    let matrixB = [Float16](unsafeUninitializedCapacity: K * N) { buffer, initializedCount in
        for i in 0..<(K * N) { buffer[i] = Float16.random(in: 0...1) }
        initializedCount = K * N
    }

    bufferA.contents().copyMemory(from: matrixA, byteCount: sizeA)
    bufferB.contents().copyMemory(from: matrixB, byteCount: sizeB)
    bufferC.contents().initializeMemory(as: UInt8.self, repeating: 0, count: sizeC)

    return (bufferA, bufferB, bufferC, matrixA, matrixB)
}

func setupVectorMatrixTest(device: MTLDevice, K: Int, N: Int) throws -> (bufferA: MTLBuffer, bufferB: MTLBuffer, bufferC: MTLBuffer, vectorA: [Float16], matrixB: [Float16]) {
    let sizeA = 1 * K * MemoryLayout<Float16>.size
    let sizeB = K * N * MemoryLayout<Float16>.size
    let sizeC = 1 * N * MemoryLayout<Float16>.size

    guard let bufferA = device.makeBuffer(length: sizeA, options: .storageModeShared),
          let bufferB = device.makeBuffer(length: sizeB, options: .storageModeShared),
          let bufferC = device.makeBuffer(length: sizeC, options: .storageModeShared) else {
        throw TestError("Could not create buffers for Vector-Matrix test")
    }

    var vectorA = [Float16](unsafeUninitializedCapacity: K) { $1 = K }
    for i in 0..<K { vectorA[i] = Float16.random(in: 0...1) }
    
    var matrixB = [Float16](unsafeUninitializedCapacity: K * N) { $1 = K * N }
    for i in 0..<(K * N) { matrixB[i] = Float16.random(in: 0...1) }

    bufferA.contents().copyMemory(from: vectorA, byteCount: sizeA)
    bufferB.contents().copyMemory(from: matrixB, byteCount: sizeB)
    bufferC.contents().initializeMemory(as: UInt8.self, repeating: 0, count: sizeC)
    
    return (bufferA, bufferB, bufferC, vectorA, matrixB)
}

func setupMatrixVectorTest(device: MTLDevice, M: Int, K: Int) throws -> (bufferA: MTLBuffer, bufferB: MTLBuffer, bufferC: MTLBuffer, matrixA: [Float16], vectorB: [Float16]) {
    let sizeA = M * K * MemoryLayout<Float16>.size
    let sizeB = K * 1 * MemoryLayout<Float16>.size
    let sizeC = M * 1 * MemoryLayout<Float16>.size

    guard let bufferA = device.makeBuffer(length: sizeA, options: .storageModeShared),
          let bufferB = device.makeBuffer(length: sizeB, options: .storageModeShared),
          let bufferC = device.makeBuffer(length: sizeC, options: .storageModeShared) else {
        throw TestError("Could not create buffers for Matrix-Vector test")
    }

    var matrixA = [Float16](unsafeUninitializedCapacity: M * K) { $1 = M * K }
    for i in 0..<(M * K) { matrixA[i] = Float16.random(in: 0...1) }
    
    var vectorB = [Float16](unsafeUninitializedCapacity: K) { $1 = K }
    for i in 0..<K { vectorB[i] = Float16.random(in: 0...1) }

    bufferA.contents().copyMemory(from: matrixA, byteCount: sizeA)
    bufferB.contents().copyMemory(from: vectorB, byteCount: sizeB)
    bufferC.contents().initializeMemory(as: UInt8.self, repeating: 0, count: sizeC)
    
    return (bufferA, bufferB, bufferC, matrixA, vectorB)
}

func executeAndWait(
    context: TestContext,
    testName: String,
    encodeCommands: (MTL4CommandBuffer) throws -> Void
) async throws {
    guard let commandBuffer = context.device.makeCommandBuffer() else {
        throw TestError("Could not create command buffer")
    }
    
    commandBuffer.beginCommandBuffer(allocator: context.commandAllocator)
    let start = ContinuousClock.now
    
    try encodeCommands(commandBuffer)
    
    commandBuffer.endCommandBuffer()
    
    let commitOptions = MTL4CommitOptions()
    try await withCheckedThrowingContinuation { (cont: CheckedContinuation<Void, Error>) in
        commitOptions.addFeedbackHandler { feedback in
            if let error = feedback.error {
                cont.resume(throwing: TestError("GPU execution failed for \(testName): \(error)"))
            } else {
                cont.resume()
            }
        }
        context.commandQueue.commit([commandBuffer], options: commitOptions)
    }

    let elapsed = start.duration(to: ContinuousClock.now)
    print("GPU time (\(testName)): \(Double(elapsed.components.seconds) * 1000 + Double(elapsed.components.attoseconds) / 1e15) ms")
}

func verifyMatrixMatrixResult(
    bufferC: MTLBuffer, M: Int, N: Int, K: Int,
    matrixA: [Float16], matrixB: [Float16],
    epsilon: Float = 1e-1
) {
    let resultPtr = bufferC.contents().bindMemory(to: Float16.self, capacity: M * N)
    let resultMatrix = Array(UnsafeBufferPointer(start: resultPtr, count: M * N))

    let testIndices = [(0,0), (M-1,N-1), (M/2,N/2)]
    for (row, col) in testIndices {
        var expected: Float = 0
        for k in 0..<K {
            expected += Float(matrixA[row * K + k]) * Float(matrixB[k * N + col])
        }
        let gpuValue = Float(resultMatrix[row * N + col])
        print("C[\(row),\(col)]: GPU=\(gpuValue), CPU=\(expected)")
        #expect(abs(gpuValue - expected) < epsilon, "Mismatch at (\(row),\(col))")
    }
}

func verifyVectorMatrixResult(
    bufferC: MTLBuffer, N: Int, K: Int,
    vectorA: [Float16], matrixB: [Float16],
    epsilon: Float = 1e-1
) {
    let resultPtr = bufferC.contents().bindMemory(to: Float16.self, capacity: N)
    let resultVector = Array(UnsafeBufferPointer(start: resultPtr, count: N))

    for n in [0, N/2, N-1] {
        var expected: Float = 0
        for k in 0..<K {
            expected += Float(vectorA[k]) * Float(matrixB[k * N + n])
        }
        let gpuValue = Float(resultVector[n])
        print("C[0,\(n)]: GPU=\(gpuValue), CPU=\(expected)")
        #expect(abs(gpuValue - expected) < epsilon, "Mismatch at (0,\(n))")
    }
}

func verifyMatrixVectorResult(
    bufferC: MTLBuffer, M: Int, K: Int,
    matrixA: [Float16], vectorB: [Float16],
    epsilon: Float = 1e-1
) {
    let resultPtr = bufferC.contents().bindMemory(to: Float16.self, capacity: M)
    let resultVector = Array(UnsafeBufferPointer(start: resultPtr, count: M))

    for m in [0, M/2, M-1] {
        var expected: Float = 0
        for k in 0..<K {
            expected += Float(matrixA[m * K + k]) * Float(vectorB[k])
        }
        let gpuValue = Float(resultVector[m])
        print("C[\(m),0]: GPU=\(gpuValue), CPU=\(expected)")
        #expect(abs(gpuValue - expected) < epsilon, "Mismatch at (\(m),0)")
    }
}
