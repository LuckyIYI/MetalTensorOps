import Foundation
import Testing
import Metal

struct Metal3Tests {
    @Test func testSimdgroupMatmulMetal3() async throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw TestError("Metal is not supported on this device")
        }
        guard let commandQueue = device.makeCommandQueue() else {
            throw TestError("Failed to create command queue")
        }
        
        let M = 128
        let N = 64
        let K = 256
        
        let sizeA = M * K * MemoryLayout<Float16>.size
        let sizeB = K * N * MemoryLayout<Float16>.size
        let sizeC = M * N * MemoryLayout<Float32>.size
        
        guard let bufferA = device.makeBuffer(length: sizeA, options: .storageModeShared) else {
            throw TestError("Failed to create buffer A")
        }
        guard let bufferB = device.makeBuffer(length: sizeB, options: .storageModeShared) else {
            throw TestError("Failed to create buffer B")
        }
        guard let bufferC = device.makeBuffer(length: sizeC, options: .storageModeShared) else {
            throw TestError("Failed to create buffer C")
        }
        
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
        
        let library = try device.makeDefaultLibrary(bundle: .main)
        let encoder = try SimdMatrixMatrixMetal3Encoder(device: device, library: library, bufferA: bufferA, bufferB: bufferB, bufferC: bufferC)
        
        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            throw TestError("Failed to create command buffer")
        }
        
        let start = ContinuousClock.now
        
        try encoder.encode(commandBuffer: commandBuffer, M: M, N: N, K: K)
        
        commandBuffer.commit()
        await commandBuffer.completed()
        
        let elapsed = start.duration(to: ContinuousClock.now)
        print("GPU time (SimdgroupMatmulMetal3): \(Double(elapsed.components.seconds) * 1000 + Double(elapsed.components.attoseconds) / 1e15) ms")
        
        let resultPtr = bufferC.contents().bindMemory(to: Float32.self, capacity: M * N)
        let resultMatrix = Array(UnsafeBufferPointer(start: resultPtr, count: M * N))
        
//        print("Result matrix (first 10 elements): \(resultMatrix.prefix(10))")
        
        let epsilon: Float = 1e-1
        for row in [0, M/2, M-1] {
            for col in [0, N/2, N-1] {
                var expected: Float = 0
                for k in 0..<K {
                    expected += Float(matrixA[row * K + k]) * Float(matrixB[k * N + col])
                }
                let gpuValue = Float(resultMatrix[row * N + col])
                print("C[\(row),\(col)]: expected = \(expected), real = \(gpuValue)")
                #expect(abs(gpuValue - expected) < epsilon, "Mismatch at (\(row),\(col)): CPU = \(expected), GPU = \(gpuValue)")
            }
        }
    }
}
