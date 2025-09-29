import Foundation
import Testing
import Metal

// Deterministic data generators so weights scale with requested dimensions.
private func makeTestVector(length: Int, seed: Float) -> [Float] {
    (0..<length).map { index in
        let x = seed + Float(index) * 0.613
        let primary = sinf(x) * 0.35
        let secondary = cosf(x * 0.5) * 0.15
        return primary + secondary
    }
}

private func makeTestMatrix(rows: Int, columns: Int, seed: Float) -> [[Float]] {
    (0..<rows).map { r in
        (0..<columns).map { c in
            let x = seed + Float(r) * 0.37 + Float(c) * 0.19
            let primary = sinf(x) * 0.3
            let secondary = cosf(x * 0.41) * 0.2
            return primary + secondary
        }
    }
}

// MARK: - Metal 4 Tests

struct Metal4Tests {
    @Test func testThreadMLP() async throws {
        let context = try TestContext()

        let inputDim = 4
        let hiddenDim = 256
        let outputDim = 8

        let inputFloats = makeTestVector(length: inputDim, seed: 0.25)
        let weightsInputHidden = makeTestMatrix(rows: hiddenDim, columns: inputDim, seed: 1.1)
        let weightsHiddenOutput = makeTestMatrix(rows: outputDim, columns: hiddenDim, seed: 2.7)
        let biasHidden = makeTestVector(length: hiddenDim, seed: 3.5)
        let biasOutput = makeTestVector(length: outputDim, seed: 4.3)

        let inputHalf = inputFloats.map(Float16.init)
        func flattenColumnMajor(_ matrix: [[Float]]) -> [Float16] {
            let rows = matrix.count
            let cols = matrix.first?.count ?? 0
            var flattened = [Float16]()
            flattened.reserveCapacity(rows * cols)
            for c in 0..<cols {
                for r in 0..<rows {
                    flattened.append(Float16(matrix[r][c]))
                }
            }
            return flattened
        }

        let weightsIHFlat = flattenColumnMajor(weightsInputHidden)
        let weightsHOFlat = flattenColumnMajor(weightsHiddenOutput)
        let biasHiddenHalf = biasHidden.map(Float16.init)
        let biasOutputHalf = biasOutput.map(Float16.init)

        guard
            let inputBuffer = context.device.makeBuffer(bytes: inputHalf, length: inputHalf.count * MemoryLayout<Float16>.stride, options: .storageModeShared),
            let weightsIHBuffer = context.device.makeBuffer(bytes: weightsIHFlat, length: weightsIHFlat.count * MemoryLayout<Float16>.stride, options: .storageModeShared),
            let weightsHOBuffer = context.device.makeBuffer(bytes: weightsHOFlat, length: weightsHOFlat.count * MemoryLayout<Float16>.stride, options: .storageModeShared),
            let biasHiddenBuffer = context.device.makeBuffer(bytes: biasHiddenHalf, length: biasHiddenHalf.count * MemoryLayout<Float16>.stride, options: .storageModeShared),
            let biasOutputBuffer = context.device.makeBuffer(bytes: biasOutputHalf, length: biasOutputHalf.count * MemoryLayout<Float16>.stride, options: .storageModeShared),
            let outputBuffer = context.device.makeBuffer(length: outputDim * MemoryLayout<Float16>.stride, options: .storageModeShared)
        else {
            throw TestError("Failed to allocate buffers for two-layer MLP test")
        }

        outputBuffer.contents().initializeMemory(as: UInt8.self, repeating: 0, count: outputDim * MemoryLayout<Float16>.stride)

        let tensorWeightsIH = try makeTensor(from: weightsIHBuffer, rows: hiddenDim, columns: inputDim, layout: .rowMajor)
        let tensorWeightsHO = try makeTensor(from: weightsHOBuffer, rows: outputDim, columns: hiddenDim, layout: .rowMajor)
        let tensorInput = try makeTensor(from: inputBuffer, rows: inputDim, columns: 1, layout: .rowMajor)
        let tensorOutput = try makeTensor(from: outputBuffer, rows: outputDim, columns: 1, layout: .rowMajor)

        let tensorBiasHidden = try makeVectorTensor(from: biasHiddenBuffer, length: hiddenDim)
        let tensorBiasOutput = try makeVectorTensor(from: biasOutputBuffer, length: outputDim)

        let encoder = try ThreadMLPEncoder(
            device: context.device,
            library: context.library,
            compiler: context.compiler,
            queue: context.commandQueue,
            weightsInputHidden: tensorWeightsIH,
            weightsHiddenOutput: tensorWeightsHO,
            input: tensorInput,
            biasHidden: tensorBiasHidden,
            biasOutput: tensorBiasOutput,
            output: tensorOutput,
            inputDim: UInt32(inputDim),
            hiddenDim: UInt32(hiddenDim),
            outputDim: UInt32(outputDim)
        )

        try await executeAndWait(context: context, testName: "ThreadMLP") { commandBuffer in
            encoder.encode(commandBuffer: commandBuffer)
        }

        let resultPtr = outputBuffer.contents().bindMemory(to: Float16.self, capacity: outputDim)
        let gpuOutput = (0..<outputDim).map { Float(resultPtr[$0]) }
        print("GPU output (float):", gpuOutput)

        var hiddenActivations = [Float](repeating: 0, count: hiddenDim)
        for h in 0..<hiddenDim {
            var acc = biasHidden[h]
            for i in 0..<inputDim {
                acc += weightsInputHidden[h][i] * inputFloats[i]
            }
            hiddenActivations[h] = max(acc, 0)
        }

        var cpuOutput = [Float](repeating: 0, count: outputDim)
        for o in 0..<outputDim {
            var acc = biasOutput[o]
            for h in 0..<hiddenDim {
                acc += weightsHiddenOutput[o][h] * hiddenActivations[h]
            }
            cpuOutput[o] = acc
        }

        for idx in 0..<outputDim {
            let diff = abs(gpuOutput[idx] - cpuOutput[idx])
            print("Index \(idx) -> GPU=\(gpuOutput[idx]) CPU=\(cpuOutput[idx]) diff=\(diff)")
            #expect(diff < 5e-2, "Mismatch at index \(idx) (GPU=\(gpuOutput[idx]), CPU=\(cpuOutput[idx]), diff=\(diff))")
        }
    }

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

    @Test(.timeLimit(.minutes(1))) func testThreadDynamicMLP() async throws {
        let context = try TestContext()

        // Architecture: INPUT_DIM -> HIDDEN_DIM x hiddenLayerCount -> OUTPUT_DIM
        let inputDim: UInt32 = 4
        let hiddenDim: UInt32 = 32
        let outputDim: UInt32 = 8
        let hiddenLayerCount: UInt32 = 2  // 2 hidden layers

        // Total layers = hiddenLayerCount + 1 (output layer)
        // Layer 0: input(4) -> hidden(32)
        // Layer 1: hidden(32) -> hidden(32)
        // Layer 2: hidden(32) -> output(8)
        let totalLayers = Int(hiddenLayerCount + 1)

        // Create test data for each layer
        var allWeights: [[[Float]]] = []
        var allBiases: [[Float]] = []

        for i in 0..<totalLayers {
            let isFirst = (i == 0)
            let isLast = (i == totalLayers - 1)

            let inDim = isFirst ? Int(inputDim) : Int(hiddenDim)
            let outDim = isLast ? Int(outputDim) : Int(hiddenDim)

            allWeights.append(makeTestMatrix(rows: outDim, columns: inDim, seed: Float(i) * 1.5 + 1.0))
            allBiases.append(makeTestVector(length: outDim, seed: Float(i) * 2.0 + 3.0))
        }

        let inputFloats = makeTestVector(length: Int(inputDim), seed: 0.25)
        let inputHalf = inputFloats.map(Float16.init)

        func flattenColumnMajor(_ matrix: [[Float]]) -> [Float16] {
            let rows = matrix.count
            let cols = matrix.first?.count ?? 0
            var flattened = [Float16]()
            flattened.reserveCapacity(rows * cols)
            for c in 0..<cols {
                for r in 0..<rows {
                    flattened.append(Float16(matrix[r][c]))
                }
            }
            return flattened
        }

        // Create buffers for weights and biases
        var weightBuffers: [MTLBuffer] = []
        var biasBuffers: [MTLBuffer] = []
        var weightTensors: [MTLTensor] = []
        var biasTensors: [MTLTensor] = []

        for i in 0..<totalLayers {
            let isFirst = (i == 0)
            let isLast = (i == totalLayers - 1)

            let inDim = isFirst ? Int(inputDim) : Int(hiddenDim)
            let outDim = isLast ? Int(outputDim) : Int(hiddenDim)

            let weightsFlat = flattenColumnMajor(allWeights[i])
            let biasHalf = allBiases[i].map(Float16.init)

            guard let wBuf = context.device.makeBuffer(bytes: weightsFlat, length: weightsFlat.count * MemoryLayout<Float16>.stride, options: .storageModeShared),
                  let bBuf = context.device.makeBuffer(bytes: biasHalf, length: biasHalf.count * MemoryLayout<Float16>.stride, options: .storageModeShared) else {
                throw TestError("Failed to allocate buffer for layer \(i)")
            }

            weightBuffers.append(wBuf)
            biasBuffers.append(bBuf)

            let wTensor = try makeTensor(from: wBuf, rows: outDim, columns: inDim, layout: .rowMajor)
            let bTensor = try makeVectorTensor(from: bBuf, length: outDim)

            weightTensors.append(wTensor)
            biasTensors.append(bTensor)
        }

        // Create input/output buffers
        guard let inputBuffer = context.device.makeBuffer(bytes: inputHalf, length: inputHalf.count * MemoryLayout<Float16>.stride, options: .storageModeShared),
              let outputBuffer = context.device.makeBuffer(length: Int(outputDim) * MemoryLayout<Float16>.stride, options: .storageModeShared) else {
            throw TestError("Failed to allocate input/output buffers")
        }

        outputBuffer.contents().initializeMemory(as: UInt8.self, repeating: 0, count: Int(outputDim) * MemoryLayout<Float16>.stride)

        let tensorInput = try makeTensor(from: inputBuffer, rows: Int(inputDim), columns: 1, layout: .rowMajor)
        let tensorOutput = try makeTensor(from: outputBuffer, rows: Int(outputDim), columns: 1, layout: .rowMajor)

        let encoder = try ThreadDynamicMLPEncoder(
            device: context.device,
            library: context.library,
            compiler: context.compiler,
            queue: context.commandQueue,
            weightTensors: weightTensors,
            biasTensors: biasTensors,
            input: tensorInput,
            output: tensorOutput,
            inputDim: inputDim,
            hiddenDim: hiddenDim,
            outputDim: outputDim,
            hiddenLayerCount: hiddenLayerCount
        )

        // Execute with timeout
        print("Running dynamic MLP test: \(inputDim) -> [\(hiddenDim) x \(hiddenLayerCount)] -> \(outputDim)")
        try await executeAndWait(context: context, testName: "ThreadDynamicMLP") { commandBuffer in
            encoder.encode(commandBuffer: commandBuffer)
        }

        // Verify results
        let resultPtr = outputBuffer.contents().bindMemory(to: Float16.self, capacity: Int(outputDim))
        let gpuOutput = (0..<Int(outputDim)).map { Float(resultPtr[$0]) }
        print("GPU output (float):", gpuOutput)

        // CPU reference computation
        var currentActivations = inputFloats
        for i in 0..<totalLayers {
            let isLast = (i == totalLayers - 1)
            let outDim = isLast ? Int(outputDim) : Int(hiddenDim)
            var nextActivations = [Float](repeating: 0, count: outDim)

            for o in 0..<outDim {
                var acc = allBiases[i][o]
                for j in 0..<currentActivations.count {
                    acc += allWeights[i][o][j] * currentActivations[j]
                }
                // ReLU for hidden layers, linear for output
                nextActivations[o] = isLast ? acc : max(acc, 0)
            }

            currentActivations = nextActivations
        }

        // Compare
        for idx in 0..<Int(outputDim) {
            let diff = abs(gpuOutput[idx] - currentActivations[idx])
            print("Index \(idx) -> GPU=\(gpuOutput[idx]) CPU=\(currentActivations[idx]) diff=\(diff)")
            #expect(diff < 5e-2, "Mismatch at index \(idx) (GPU=\(gpuOutput[idx]), CPU=\(currentActivations[idx]), diff=\(diff))")
        }
    }
}
