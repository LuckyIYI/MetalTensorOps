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

    @Test func testSimdgroupMatrixVector() async throws {
        let context = try TestContext()
        let problemSizes: [(rows: Int, cols: Int)] = [
            (rows: 24, cols: 37),   // small, odd sizes
            (rows: 64, cols: 128),  // power-of-two tile aligned
            (rows: 113, cols: 255)  // large, non-aligned to tile width
        ]

        for (index, problem) in problemSizes.enumerated() {
            let (bufferA, bufferB, bufferC, matrixA, vectorB) = try setupMatrixVectorTest(device: context.device, M: problem.rows, K: problem.cols)

            let tensorA = try makeTensor(from: bufferA, rows: problem.rows, columns: problem.cols)
            let tensorB = try makeTensor(from: bufferB, rows: problem.cols, columns: 1)
            let tensorC = try makeTensor(from: bufferC, rows: problem.rows, columns: 1)

            let simdgroupEncoder = try SimdMatrixVectorEncoder(
                device: context.device,
                library: context.library,
                compiler: context.compiler,
                queue: context.commandQueue,
                tensorA: tensorA,
                tensorB: tensorB,
                tensorC: tensorC
            )

            try await executeAndWait(context: context, testName: "SimdgroupMatrixVector_case\(index)") { commandBuffer in
                simdgroupEncoder.encode(commandBuffer: commandBuffer, rows: problem.rows)
            }

            verifyMatrixVectorResult(bufferC: bufferC, M: problem.rows, K: problem.cols, matrixA: matrixA, vectorB: vectorB)
        }
    }

    @Test func testSimdHierarchicalMatrixVector() async throws {
        let context = try TestContext()
        let problemSizes: [(rows: Int, cols: Int)] = [
            (rows: 24, cols: 37),
            (rows: 64, cols: 128),
            (rows: 113, cols: 255)
        ]

        for (index, problem) in problemSizes.enumerated() {
            let (bufferA, bufferB, bufferC, matrixA, vectorB) = try setupMatrixVectorTest(device: context.device, M: problem.rows, K: problem.cols)

            let hierarchicalEncoder = try SimdHierarchicalMatrixVectorEncoder(
                device: context.device,
                library: context.library,
                compiler: context.compiler,
                queue: context.commandQueue,
                bufferA: bufferA,
                bufferB: bufferB,
                bufferC: bufferC
            )

            try await executeAndWait(context: context, testName: "SimdHierarchicalMatrixVector_case\(index)") { commandBuffer in
                hierarchicalEncoder.encode(commandBuffer: commandBuffer, rows: problem.rows, columns: problem.cols)
            }

            verifyMatrixVectorResult(bufferC: bufferC, M: problem.rows, K: problem.cols, matrixA: matrixA, vectorB: vectorB)
        }
    }

    @Test(.timeLimit(.minutes(4))) func testGemvPerformance() async throws {
        let context = try TestContext()
        let cases: [(rows: Int, cols: Int, samples: Int)] = [
            (rows: 128, cols: 512, samples: 8),
            (rows: 256, cols: 1024, samples: 10),
            (rows: 384, cols: 1536, samples: 10),
            (rows: 512, cols: 2048, samples: 12),
            (rows: 640, cols: 1024, samples: 10),
            (rows: 768, cols: 2560, samples: 12),
            (rows: 896, cols: 1792, samples: 12),
            (rows: 1024, cols: 3072, samples: 14),
            (rows: 1280, cols: 4096, samples: 14),
            (rows: 1536, cols: 512, samples: 10),
            (rows: 1792, cols: 1536, samples: 12),
            (rows: 2048, cols: 4096, samples: 16)
        ]

        struct PerfResult {
            let rows: Int
            let cols: Int
            let simdgroupAverage: Double
            let hierarchicalAverage: Double
            let maxULPDiff: Int
        }

        var results: [PerfResult] = []

        for (rows, cols, iterations) in cases {
            print("== GEMV performance case: rows=\(rows) cols=\(cols) ==")

            let warmupRuns = max(2, iterations / 4)

            let (bufferA, bufferB, bufferCTensor, matrixA, vectorB) = try setupMatrixVectorTest(device: context.device, M: rows, K: cols)

            guard let bufferCSimd = context.device.makeBuffer(length: rows * MemoryLayout<Float16>.stride, options: .storageModeShared) else {
                throw TestError("Could not allocate SIMD GEMV output buffer")
            }
            bufferCSimd.contents().initializeMemory(as: UInt8.self, repeating: 0, count: rows * MemoryLayout<Float16>.stride)

            let tensorA = try makeTensor(from: bufferA, rows: rows, columns: cols)
            let tensorB = try makeTensor(from: bufferB, rows: cols, columns: 1)
            let tensorC = try makeTensor(from: bufferCTensor, rows: rows, columns: 1)

            let simdgroupEncoder = try SimdMatrixVectorEncoder(
                device: context.device,
                library: context.library,
                compiler: context.compiler,
                queue: context.commandQueue,
                tensorA: tensorA,
                tensorB: tensorB,
                tensorC: tensorC
            )

            let hierarchicalEncoder = try SimdHierarchicalMatrixVectorEncoder(
                device: context.device,
                library: context.library,
                compiler: context.compiler,
                queue: context.commandQueue,
                bufferA: bufferA,
                bufferB: bufferB,
                bufferC: bufferCSimd
            )

            func measureAverage(label: String, encode: @escaping (MTL4CommandBuffer) throws -> Void) async throws -> Double {
                var totalMilliseconds: Double = 0
                for run in 0..<iterations {
                    guard let commandBuffer = context.device.makeCommandBuffer() else {
                        throw TestError("Failed to create command buffer for \(label)")
                    }

                    commandBuffer.beginCommandBuffer(allocator: context.commandAllocator)
                    let start = ContinuousClock.now
                    try encode(commandBuffer)
                    commandBuffer.endCommandBuffer()

                    let commitOptions = MTL4CommitOptions()
                    try await withCheckedThrowingContinuation { (cont: CheckedContinuation<Void, Error>) in
                        commitOptions.addFeedbackHandler { feedback in
                            if let error = feedback.error {
                                cont.resume(throwing: TestError("GPU execution failed for \(label): \(error)"))
                            } else {
                                cont.resume()
                            }
                        }
                        context.commandQueue.commit([commandBuffer], options: commitOptions)
                    }

                    let elapsed = start.duration(to: ContinuousClock.now)
                    let milliseconds = Double(elapsed.components.seconds) * 1_000.0
                        + Double(elapsed.components.attoseconds) / 1_000_000_000_000_000.0
                    if run >= warmupRuns {
                        totalMilliseconds += milliseconds
                    }
                }

                let samples = iterations - warmupRuns
                let average = totalMilliseconds / Double(samples)
                print("\(label) average over \(samples) runs: \(average) ms")
                return average
            }

            let simdgroupAverage = try await measureAverage(label: "Tensor SIMDGROUP GEMV") { commandBuffer in
                simdgroupEncoder.encode(commandBuffer: commandBuffer, rows: rows)
            }

            let hierarchicalAverage = try await measureAverage(label: "SIMD Hierarchical GEMV") { commandBuffer in
                hierarchicalEncoder.encode(commandBuffer: commandBuffer, rows: rows, columns: cols)
            }

            verifyMatrixVectorResult(bufferC: bufferCTensor, M: rows, K: cols, matrixA: matrixA, vectorB: vectorB, epsilon: 5e-1)
            verifyMatrixVectorResult(bufferC: bufferCSimd, M: rows, K: cols, matrixA: matrixA, vectorB: vectorB, epsilon: 5e-1)

            let tensorPtr = bufferCTensor.contents().bindMemory(to: Float16.self, capacity: rows)
            let simdPtr = bufferCSimd.contents().bindMemory(to: Float16.self, capacity: rows)
            var maxULPDiff = 0
            for index in 0..<rows {
                let ulpDiff = abs(Int(tensorPtr[index].bitPattern) - Int(simdPtr[index].bitPattern))
                if ulpDiff > maxULPDiff {
                    maxULPDiff = ulpDiff
                }
            }
            print("Max ULP difference between SIMDGROUP and hierarchical GEMV results: \(maxULPDiff)")
            #expect(maxULPDiff <= 1, "Hierarchical GEMV result deviates from SIMDGROUP result (max ULP diff \(maxULPDiff))")

            results.append(PerfResult(rows: rows, cols: cols, simdgroupAverage: simdgroupAverage, hierarchicalAverage: hierarchicalAverage, maxULPDiff: maxULPDiff))
        }

        print("== GEMV Performance CSV ==")
        print("rows,cols,tensorSIMD_ms,hierarchicalSIMD_ms,ulpDiff")
        for result in results {
            print("\(result.rows),\(result.cols),\(result.simdgroupAverage),\(result.hierarchicalAverage),\(result.maxULPDiff)")
        }

        let csvLines = results.map { result in
            "\(result.rows),\(result.cols),\(result.simdgroupAverage),\(result.hierarchicalAverage),\(result.maxULPDiff)"
        }
        let csv = ([ "rows,cols,tensorSIMD_ms,hierarchicalSIMD_ms,ulpDiff" ] + csvLines).joined(separator: "\n") + "\n"

        do {
            var rootURL = URL(fileURLWithPath: #filePath)
            while rootURL.lastPathComponent != "MetalTensorOpTests" && rootURL.pathComponents.count > 1 {
                rootURL.deleteLastPathComponent()
            }
            if rootURL.lastPathComponent == "MetalTensorOpTests" {
                rootURL.deleteLastPathComponent()
            }
            let outputDirectory = rootURL.appendingPathComponent("PerfResults", isDirectory: true)
            try FileManager.default.createDirectory(at: outputDirectory, withIntermediateDirectories: true)
            let timestamp = ISO8601DateFormatter().string(from: Date()).replacingOccurrences(of: ":", with: "-")
            let outputURL = outputDirectory.appendingPathComponent("gemv_performance_\(timestamp).csv")
            try csv.write(to: outputURL, atomically: true, encoding: .utf8)
            print("Wrote GEMV performance CSV to \(outputURL.path)")
        } catch {
            print("Failed to write GEMV performance CSV: \(error)")
        }
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
