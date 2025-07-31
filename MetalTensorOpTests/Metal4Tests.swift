import Foundation
import Testing
import Metal

// MARK: - Metal 4 Tests

struct Metal4Tests {
    @Test func testSimdgroupMatrixMatrix() async throws {
        let context = try TestContext()
        let M = 128, N = 64, K = 256

        let (bufferA, bufferB, bufferC, matrixA, matrixB) = try setupMatrixMatrixTest(device: context.device, M: M, N: N, K: K)

        let tensorA = try makeTensor(from: bufferA, rows: M, columns: K)
        let tensorB = try makeTensor(from: bufferB, rows: K, columns: N)
        let tensorC = try makeTensor(from: bufferC, rows: M, columns: N)

        let encoder = try SimdMatrixMatrixEncoder(
            device: context.device, library: context.library, compiler: context.compiler,
            queue: context.commandQueue, tensorA: tensorA, tensorB: tensorB, tensorC: tensorC
        )

        try await executeAndWait(context: context, testName: "SimdgroupMatmul") { commandBuffer in
            encoder.encode(commandBuffer: commandBuffer, M: M, N: N)
        }

        verifyMatrixMatrixResult(bufferC: bufferC, M: M, N: N, K: K, matrixA: matrixA, matrixB: matrixB)
    }

    @Test func testThreadMatrixMatrix() async throws {
        let context = try TestContext()
        let M = 64, N = 32, K = 32

        let (bufferA, bufferB, bufferC, matrixA, matrixB) = try setupMatrixMatrixTest(device: context.device, M: M, N: N, K: K)

        let tensorA = try makeTensor(from: bufferA, rows: M, columns: K)
        let tensorB = try makeTensor(from: bufferB, rows: K, columns: N)
        let tensorC = try makeTensor(from: bufferC, rows: M, columns: N)

        let encoder = try ThreadMatrixMatrixEncoder(
            device: context.device, library: context.library, compiler: context.compiler,
            queue: context.commandQueue, tensorA: tensorA, tensorB: tensorB, tensorC: tensorC
        )

        try await executeAndWait(context: context, testName: "ThreadMatmul") { commandBuffer in
            encoder.encode(commandBuffer: commandBuffer)
        }

        verifyMatrixMatrixResult(bufferC: bufferC, M: M, N: N, K: K, matrixA: matrixA, matrixB: matrixB)
    }

    @Test func testThreadVectorMatix() async throws {
        let context = try TestContext()
        let K = 31, N = 17

        let (bufferA, bufferB, bufferC, vectorA, matrixB) = try setupVectorMatrixTest(device: context.device, K: K, N: N)

        let tensorA = try makeTensor(from: bufferA, rows: 1, columns: K)
        let tensorB = try makeTensor(from: bufferB, rows: K, columns: N)
        let tensorC = try makeTensor(from: bufferC, rows: 1, columns: N)

        let encoder = try ThreadVectorMatrixEncoder(
            device: context.device, library: context.library, compiler: context.compiler,
            queue: context.commandQueue, tensorA: tensorA, tensorB: tensorB, tensorC: tensorC
        )

        try await executeAndWait(context: context, testName: "ThreadVectorMatrix") { commandBuffer in
            encoder.encode(commandBuffer: commandBuffer)
        }

        verifyVectorMatrixResult(bufferC: bufferC, N: N, K: K, vectorA: vectorA, matrixB: matrixB)
    }
    
    @Test func testThreadMatrixVector() async throws {
        let context = try TestContext()
        let M = 29, K = 19

        let (bufferA, bufferB, bufferC, matrixA, vectorB) = try setupMatrixVectorTest(device: context.device, M: M, K: K)

        let tensorA = try makeTensor(from: bufferA, rows: M, columns: K)
        let tensorB = try makeTensor(from: bufferB, rows: K, columns: 1)
        let tensorC = try makeTensor(from: bufferC, rows: M, columns: 1)

        let encoder = try ThreadMatrixVectorEncoder(
            device: context.device, library: context.library, compiler: context.compiler,
            queue: context.commandQueue, tensorA: tensorA, tensorB: tensorB, tensorC: tensorC
        )

        try await executeAndWait(context: context, testName: "ThreadMatrixVector") { commandBuffer in
            encoder.encode(commandBuffer: commandBuffer)
        }

        verifyMatrixVectorResult(bufferC: bufferC, M: M, K: K, matrixA: matrixA, vectorB: vectorB)
    }
}
