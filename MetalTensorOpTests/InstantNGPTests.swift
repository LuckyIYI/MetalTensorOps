import Foundation
import Testing
import Metal
@testable import MetalTensorOp
import simd

private var instantNGPLogBuffer: [String] = []


private let instantNGPLogURL: URL = {
    let base = URL(fileURLWithPath: NSTemporaryDirectory(), isDirectory: true)
    return base.appendingPathComponent("instant_ngp_test.log")
}()

@inline(__always)
private func logLine(_ text: String) {
    instantNGPLogBuffer.append(text)

    guard let data = (text + "\n").data(using: .utf8) else { return }
    // Mirror output to stderr (useful when the harness streams it)
    FileHandle.standardError.write(data)

    if FileManager.default.fileExists(atPath: instantNGPLogURL.path) == false {
        try? Data().write(to: instantNGPLogURL)
    }

    if let handle = try? FileHandle(forWritingTo: instantNGPLogURL) {
        defer { try? handle.close() }
        do { try handle.seekToEnd() } catch {}
        handle.write(data)
        try? handle.synchronize()
    }

    let repoLogURL = FileManager.default.homeDirectoryForCurrentUser
        .appendingPathComponent("instant_ngp_test.log")
    if FileManager.default.fileExists(atPath: repoLogURL.path) == false {
        try? Data().write(to: repoLogURL)
    }
    if let handle = try? FileHandle(forWritingTo: repoLogURL) {
        defer { try? handle.close() }
        do { try handle.seekToEnd() } catch {}
        handle.write(data)
        try? handle.synchronize()
    }
}

@MainActor
struct InstantNGPTests {
    @Test func test() async throws {

        let context = try TestContext()

        logLine("cwd: \(FileManager.default.currentDirectoryPath)")
        logLine("tmp: \(NSTemporaryDirectory())")

        // Load trained weights - find test bundle resources
        guard let testBundle = Bundle.allBundles.first(where: { $0.bundlePath.contains("MetalTensorOpTests.xctest") }) else {
            throw TestError("Test bundle not found")
        }

        guard let weightsURL = testBundle.url(forResource: "instant_ngp", withExtension: "json") else {
            throw TestError("instant_ngp.json not found - run training script first")
        }
        let weights = try InstantNGPWeightsFile.load(from: weightsURL)
        let metalWeights = try weights.makeMetalWeights(device: context.device)
        let encoding = weights.encoding

        let samples = (try? weights.makeSampleDataset()) ?? []
        if let declared = weights.sampleCount {
            #expect(declared == samples.count, "Declared sample_count (\(declared)) does not match actual count (\(samples.count))")
        }
        #expect(!samples.isEmpty, "instant_ngp.json must embed sparse samples")
        for sample in samples {
            let pos = sample.position
            let val = sample.value
            #expect(pos.x >= 0 && pos.x <= 1 && pos.y >= 0 && pos.y <= 1, "Sample positions must be in [0, 1]")
            #expect(val.x >= 0 && val.x <= 1 && val.y >= 0 && val.y <= 1 && val.z >= 0 && val.z <= 1, "Sample values must be in [0, 1]")
        }

        logLine("Loaded trained weights:")
        logLine("  Hash table: \(encoding.hash_table.shape)")
        logLine("  Image size: \(weights.metadata.image?.width ?? 0)x\(weights.metadata.image?.height ?? 0)")
        logLine("  Sample count: \(samples.count)")

        // Exercise the production encoder so the test covers the shipping path.
        let encoder = try InstantNGPEncoder(
            device: context.device,
            library: context.library,
            compiler: context.compiler,
            queue: context.commandQueue,
            weights: metalWeights
        )

        var testPositions: [Float] = []
        var expectedColors: [[Float]] = []
        for sample in samples {
            testPositions.append(sample.position.x)
            testPositions.append(sample.position.y)
            expectedColors.append([sample.value.x, sample.value.y, sample.value.z])
        }

        let cpuReference = CPUReference(model: weights)
        let (cpuEncodings, cpuHidden, cpuOutputsFlat) = cpuReference.forward(positions: testPositions)

        let numPositions = testPositions.count / 2
        let cpuOutputsStructured: [[Float]] = (0..<numPositions).map { idx in
            let start = idx * InstantNGPConfig.mlpOutputDim
            return Array(cpuOutputsFlat[start ..< start + InstantNGPConfig.mlpOutputDim])
        }
        // CPU reference matches MLX numerics; use this log for quick inspection.
        // Avoid duplicate stdout/err by only logging via logLine.
        logLine("CPU reference outputs: \(cpuOutputsStructured)")

        for (i, sample) in expectedColors.enumerated() {
            let cpuOut = cpuOutputsStructured[i]
            logLine("GT sample \(i): expected \(sample) vs CPU \(cpuOut)")
        }

        // GPU stage buffers
        let positionsBufferLength = numPositions * 2 * MemoryLayout<Float>.stride
        guard let positionsBuffer = context.device.makeBuffer(length: positionsBufferLength, options: .storageModeShared) else {
            throw TestError("Failed to allocate positions buffer")
        }
        let posPtr = positionsBuffer.contents().bindMemory(to: Float.self, capacity: numPositions * 2)
        for i in 0..<numPositions {
            posPtr[i] = testPositions[i * 2]
            posPtr[i + numPositions] = testPositions[i * 2 + 1]
        }
        let positionsDesc = MTLTensorDescriptor()
        positionsDesc.dimensions = MTLTensorExtents([numPositions, 2])!
        positionsDesc.strides = MTLTensorExtents([1, numPositions])!
        positionsDesc.usage = .compute
        positionsDesc.dataType = .float32
        let positionsTensor = try positionsBuffer.makeTensor(descriptor: positionsDesc, offset: 0)

        let outputsBufferLength = numPositions * InstantNGPConfig.mlpOutputDim * MemoryLayout<Float16>.stride
        guard let outputsBuffer = context.device.makeBuffer(length: outputsBufferLength, options: .storageModeShared) else {
            throw TestError("Failed to allocate outputs buffer")
        }
        let outputsDesc = MTLTensorDescriptor()
        outputsDesc.dimensions = MTLTensorExtents([numPositions, InstantNGPConfig.mlpOutputDim])!
        outputsDesc.strides = MTLTensorExtents([1, numPositions])!
        outputsDesc.usage = .compute
        outputsDesc.dataType = .float16
        let outputsTensor = try outputsBuffer.makeTensor(descriptor: outputsDesc, offset: 0)

        let encodedDebugBufferLength = numPositions * InstantNGPConfig.totalFeatures * MemoryLayout<Float>.stride
        guard let encodedDebugBuffer = context.device.makeBuffer(length: encodedDebugBufferLength, options: .storageModeShared) else {
            throw TestError("Failed to allocate encoded debug buffer")
        }
        let encodedDebugDesc = MTLTensorDescriptor()
        encodedDebugDesc.dimensions = MTLTensorExtents([numPositions, InstantNGPConfig.totalFeatures])!
        encodedDebugDesc.strides = MTLTensorExtents([1, numPositions])!
        encodedDebugDesc.usage = .compute
        encodedDebugDesc.dataType = .float32
        let encodedDebugTensor = try encodedDebugBuffer.makeTensor(descriptor: encodedDebugDesc, offset: 0)

        let hiddenDebugBufferLength = numPositions * InstantNGPConfig.mlpHiddenWidth * MemoryLayout<Float>.stride
        guard let hiddenDebugBuffer = context.device.makeBuffer(length: hiddenDebugBufferLength, options: .storageModeShared) else {
            throw TestError("Failed to allocate hidden debug buffer")
        }
        let hiddenDebugDesc = MTLTensorDescriptor()
        hiddenDebugDesc.dimensions = MTLTensorExtents([numPositions, InstantNGPConfig.mlpHiddenWidth])!
        hiddenDebugDesc.strides = MTLTensorExtents([1, numPositions])!
        hiddenDebugDesc.usage = .compute
        hiddenDebugDesc.dataType = .float32
        let hiddenDebugTensor = try hiddenDebugBuffer.makeTensor(descriptor: hiddenDebugDesc, offset: 0)

        let outputDebugBufferLength = numPositions * InstantNGPConfig.mlpOutputDim * MemoryLayout<Float>.stride
        guard let outputDebugBuffer = context.device.makeBuffer(length: outputDebugBufferLength, options: .storageModeShared) else {
            throw TestError("Failed to allocate output debug buffer")
        }
        let outputDebugDesc = MTLTensorDescriptor()
        outputDebugDesc.dimensions = MTLTensorExtents([numPositions, InstantNGPConfig.mlpOutputDim])!
        outputDebugDesc.strides = MTLTensorExtents([1, numPositions])!
        outputDebugDesc.usage = .compute
        outputDebugDesc.dataType = .float32
        let outputDebugTensor = try outputDebugBuffer.makeTensor(descriptor: outputDebugDesc, offset: 0)

        try await executeAndWait(context: context, testName: "InstantNGP debug") { commandBuffer in
            let residency = try context.device.makeResidencySet(descriptor: .init())
            context.commandQueue.addResidencySet(residency)
            residency.addAllocation(positionsBuffer)
            residency.addAllocation(positionsTensor)
            residency.addAllocation(outputsBuffer)
            residency.addAllocation(outputsTensor)
            residency.addAllocation(encodedDebugBuffer)
            residency.addAllocation(encodedDebugTensor)
            residency.addAllocation(hiddenDebugBuffer)
            residency.addAllocation(hiddenDebugTensor)
            residency.addAllocation(outputDebugBuffer)
            residency.addAllocation(outputDebugTensor)
            residency.commit()
            commandBuffer.useResidencySet(residency)
            encoder.encodeInferenceDebug(
                positions: positionsTensor,
                outputs: outputsTensor,
                numPositions: UInt32(numPositions),
                encodedDebug: encodedDebugTensor,
                hiddenDebug: hiddenDebugTensor,
                outputDebug: outputDebugTensor,
                commandBuffer: commandBuffer
            )
        }

        func columnMajorToRowMajor(ptr: UnsafePointer<Float>, rows: Int, cols: Int) -> [Float] {
            var result = [Float](repeating: 0, count: rows * cols)
            for r in 0..<rows {
                for c in 0..<cols {
                    let src = c * rows + r
                    result[r * cols + c] = ptr[src]
                }
            }
            return result
        }

        let gpuEncodings = columnMajorToRowMajor(ptr: encodedDebugBuffer.contents().bindMemory(to: Float.self, capacity: encodedDebugBufferLength / MemoryLayout<Float>.stride), rows: numPositions, cols: InstantNGPConfig.totalFeatures)
        let gpuHidden = columnMajorToRowMajor(ptr: hiddenDebugBuffer.contents().bindMemory(to: Float.self, capacity: hiddenDebugBufferLength / MemoryLayout<Float>.stride), rows: numPositions, cols: InstantNGPConfig.mlpHiddenWidth)
        let gpuOutputsDebug = columnMajorToRowMajor(ptr: outputDebugBuffer.contents().bindMemory(to: Float.self, capacity: outputDebugBufferLength / MemoryLayout<Float>.stride), rows: numPositions, cols: InstantNGPConfig.mlpOutputDim)

        var maxEncodingDiff: Float = 0
        for i in 0..<(numPositions * InstantNGPConfig.totalFeatures) {
            maxEncodingDiff = max(maxEncodingDiff, abs(gpuEncodings[i] - cpuEncodings[i]))
        }
        logLine(String(format: "Encoding max diff GPU vs CPU: %.6f", maxEncodingDiff))
        #expect(maxEncodingDiff < 1e-3, "Encoding diff too large: \(maxEncodingDiff)")

        var maxHiddenDiff: Float = 0
        for i in 0..<(numPositions * InstantNGPConfig.mlpHiddenWidth) {
            maxHiddenDiff = max(maxHiddenDiff, abs(gpuHidden[i] - cpuHidden[i]))
        }
        logLine(String(format: "Hidden max diff GPU vs CPU: %.6f", maxHiddenDiff))
        #expect(maxHiddenDiff < 1e-3, "Hidden activation diff too large: \(maxHiddenDiff)")

        var maxOutputDiff: Float = 0
        for i in 0..<(numPositions * InstantNGPConfig.mlpOutputDim) {
            maxOutputDiff = max(maxOutputDiff, abs(gpuOutputsDebug[i] - cpuOutputsFlat[i]))
        }
        logLine(String(format: "Output max diff GPU vs CPU: %.6f", maxOutputDiff))
        #expect(maxOutputDiff < 1e-3, "Output diff too large: \(maxOutputDiff)")

        // Final inference (half outputs)
        try await executeAndWait(context: context, testName: "InstantNGP inference") { commandBuffer in
            let residency = try context.device.makeResidencySet(descriptor: .init())
            context.commandQueue.addResidencySet(residency)
            residency.addAllocation(positionsBuffer)
            residency.addAllocation(positionsTensor)
            residency.addAllocation(outputsBuffer)
            residency.addAllocation(outputsTensor)
            residency.commit()
            commandBuffer.useResidencySet(residency)
            encoder.encodeInference(
                positions: positionsTensor,
                outputs: outputsTensor,
                numPositions: UInt32(numPositions),
                commandBuffer: commandBuffer
            )
        }

        let outputPtr = outputsBuffer.contents().bindMemory(to: Float16.self, capacity: numPositions * InstantNGPConfig.mlpOutputDim)
        var metalOutputs = [Float](repeating: 0, count: numPositions * InstantNGPConfig.mlpOutputDim)
        for pos in 0..<numPositions {
            for channel in 0..<InstantNGPConfig.mlpOutputDim {
                let srcIndex = channel * numPositions + pos
                metalOutputs[pos * InstantNGPConfig.mlpOutputDim + channel] = Float(outputPtr[srcIndex])
            }
        }

        logLine("\n✓ Metal Inference Results vs MLX (CPU reference):")
        var gtSumDiffs: [Float] = []
        for i in 0..<numPositions {
            let metalR = metalOutputs[i*3 + 0]
            let metalG = metalOutputs[i*3 + 1]
            let metalB = metalOutputs[i*3 + 2]

            let mlxR = cpuOutputsStructured[i][0]
            let mlxG = cpuOutputsStructured[i][1]
            let mlxB = cpuOutputsStructured[i][2]

            let diffR = abs(metalR - mlxR)
            let diffG = abs(metalG - mlxG)
            let diffB = abs(metalB - mlxB)

            logLine("  Position \(i): Metal[\(metalR), \(metalG), \(metalB)] vs CPU/MLX[\(mlxR), \(mlxG), \(mlxB)]")
            logLine("             Diff: [\(diffR), \(diffG), \(diffB)]")

            #expect(diffR < 0.01, "R channel diff too large: \(diffR)")
            #expect(diffG < 0.01, "G channel diff too large: \(diffG)")
            #expect(diffB < 0.01, "B channel diff too large: \(diffB)")

            let target = expectedColors[i]
            let gtDiff = abs(metalR - target[0]) + abs(metalG - target[1]) + abs(metalB - target[2])
            gtSumDiffs.append(gtDiff)
            logLine("             GT sum diff: \(gtDiff)")
        }

        // Summarize GT error to make training quality obvious when inspecting logs.
        if !gtSumDiffs.isEmpty {
            let mean = gtSumDiffs.reduce(0, +) / Float(gtSumDiffs.count)
            let maxErr = gtSumDiffs.max() ?? 0
            let minErr = gtSumDiffs.min() ?? 0
            logLine(String(format: "\nGT L1-sum error: mean=%.6f min=%.6f max=%.6f", mean, minErr, maxErr))
        }

        instantNGPLogBuffer.forEach { Swift.print($0) }
        instantNGPLogBuffer.removeAll()
    }
}



struct CPUReference {
    let totalFeatures = InstantNGPConfig.totalFeatures
    let hiddenWidth = InstantNGPConfig.mlpHiddenWidth
    let outputDim = InstantNGPConfig.mlpOutputDim
    let tableSize = 1 << InstantNGPConfig.log2HashmapSize
    let featuresPerLevel = InstantNGPConfig.featuresPerLevel
    let numLevels = InstantNGPConfig.numLevels

    let hashTable: [Float]
    let W1: [Float]
    let b1: [Float]
    let W2: [Float]
    let b2: [Float]

    // Match MLX schedule derived from config parameters
    var levelScales: [Float] {
        let lnMin = logf(Float(InstantNGPConfig.baseResolution))
        let lnMax = logf(Float(InstantNGPConfig.maxResolution))
        let levels = Float(numLevels)
        guard numLevels > 1 else { return [expf(lnMin)] }
        return (0..<numLevels).map { level in
            let t = Float(level) / (levels - 1)
            return expf(lnMin * (1 - t) + lnMax * t)
        }
    }

    init(model: InstantNGPWeightsFile) {
        // Enforce exact shapes; no tiling/padding needed in tests.
        let expectedHashCount = numLevels * tableSize * featuresPerLevel
        let rawHash = model.encoding.hash_table.data
        precondition(rawHash.count == expectedHashCount, "Hash table size mismatch: got \(rawHash.count), expected \(expectedHashCount)")
        self.hashTable = rawHash

        let L1 = model.mlp.layers[0]
        let L2 = model.mlp.layers[1]

        let expectedW1 = totalFeatures * hiddenWidth
        let expectedB1 = hiddenWidth
        let expectedW2 = hiddenWidth * outputDim
        let expectedB2 = outputDim

        precondition(L1.weights.count == expectedW1, "L1 weights size mismatch: got \(L1.weights.count), expected \(expectedW1)")
        precondition(L1.biases.count == expectedB1, "L1 bias size mismatch: got \(L1.biases.count), expected \(expectedB1)")
        precondition(L2.weights.count == expectedW2, "L2 weights size mismatch: got \(L2.weights.count), expected \(expectedW2)")
        precondition(L2.biases.count == expectedB2, "L2 bias size mismatch: got \(L2.biases.count), expected \(expectedB2)")

        self.W1 = L1.weights
        self.b1 = L1.biases
        self.W2 = L2.weights
        self.b2 = L2.biases
    }

// Forward pass on arbitrary positions list [x0,y0,x1,y1,...]
    func forward(positions: [Float]) -> (encoded: [Float], hidden: [Float], outputs: [Float]) {
        let N = positions.count / 2
        var enc = [Float](repeating: 0, count: N * totalFeatures)
        var hid = [Float](repeating: 0, count: N * hiddenWidth)
        var out = [Float](repeating: 0, count: N * outputDim)

        for i in 0..<N {
            let x = positions[i*2 + 0]
            let y = positions[i*2 + 1]
            var features = [Float](repeating: 0, count: totalFeatures)
            encodePoint(x: x, y: y, out: &features)

            for f in 0..<totalFeatures { enc[i*totalFeatures + f] = features[f] }

            // y1 = ReLU(W1 * f + b1) with half-precision multiply to match GPU
            var hidden = [Float](repeating: 0, count: hiddenWidth)
            for n in 0..<hiddenWidth {
                var sum: Float = 0
                for k in 0..<totalFeatures {
                    let f16 = Float16(features[k])
                    let w16 = Float16(W1[k*hiddenWidth + n])
                    sum += Float(w16) * Float(f16)
                }
                sum += b1[n]
                hidden[n] = max(sum, 0)
            }
            for h in 0..<hiddenWidth { hid[i*hiddenWidth + h] = hidden[h] }

            // y2 = sigmoid(W2 * y1 + b2) with half-precision multiply
            for c in 0..<outputDim {
                var sum: Float = 0
                for h in 0..<hiddenWidth {
                    let h16 = Float16(hidden[h])
                    let w16 = Float16(W2[h*outputDim + c])
                    sum += Float(w16) * Float(h16)
                }
                sum += b2[c]
                out[i*outputDim + c] = 1.0 / (1.0 + exp(-sum))
            }
        }
        return (enc, hid, out)
    }

    func forwardGrid(width w: Int, height h: Int) -> [Float] {
        var out = [Float](repeating: 0, count: w*h*3)
        for y in 0..<h {
            for x in 0..<w {
                let u = (w > 1) ? Float(x) / Float(w - 1) : 0
                let v = (h > 1) ? Float(y) / Float(h - 1) : 0
                let rgb = forwardPoint(x: u, y: v)
                let i = (y*w + x) * 3
                out[i+0] = rgb[0]
                out[i+1] = rgb[1]
                out[i+2] = rgb[2]
            }
        }
        return out
    }

    func forwardPoint(x: Float, y: Float) -> [Float] {
        var features = [Float](repeating: 0, count: totalFeatures)
        encodePoint(x: x, y: y, out: &features)
        var hidden = [Float](repeating: 0, count: hiddenWidth)
        for n in 0..<hiddenWidth {
            var sum: Float = 0
            for k in 0..<totalFeatures {
                let f16 = Float16(features[k])
                let w16 = Float16(W1[k*hiddenWidth + n])
                sum += Float(w16) * Float(f16)
            }
            sum += b1[n]
            hidden[n] = max(sum, 0)
        }
        var out = [Float](repeating: 0, count: 3)
        for c in 0..<3 {
            var sum: Float = 0
            for h in 0..<hiddenWidth {
                let h16 = Float16(hidden[h])
                let w16 = Float16(W2[h*3 + c])
                sum += Float(w16) * Float(h16)
            }
            sum += b2[c]
            out[c] = 1.0 / (1.0 + exp(-sum))
        }
        return out
    }

    func encodePoint(x: Float, y: Float, out: inout [Float]) {
        let scales = levelScales
        for level in 0..<numLevels {
            let scale = scales[level]
            let xf = x * scale
            let yf = y * scale
            let xi = floorf(xf)
            let yi = floorf(yf)
            let dx = xf - xi
            let dy = yf - yi
            let x0 = UInt32(xi)
            let y0 = UInt32(yi)
            let base = level * tableSize
            var a0: Float = 0
            var a1: Float = 0
            for corner in 0..<4 {
                let ox = UInt32((corner >> 0) & 1)
                let oy = UInt32((corner >> 1) & 1)
                let cx = x0 &+ ox
                let cy = y0 &+ oy
                let wx = (ox == 0) ? (1 - dx) : dx
                let wy = (oy == 0) ? (1 - dy) : dy
                let w = wx * wy
                let idx = hash2D(cx, cy) % UInt32(tableSize)
                let row = base + Int(idx)
                a0 += w * hashTable[row*featuresPerLevel + 0]
                a1 += w * hashTable[row*featuresPerLevel + 1]
            }
            let o = level * featuresPerLevel
            out[o+0] = a0
            out[o+1] = a1
        }
    }

    @inline(__always)
    func hash2D(_ x: UInt32, _ y: UInt32) -> UInt32 {
        let a: UInt32 = 1
        let b: UInt32 = 2654435761
        return (x &* a) ^ (y &* b)
    }
}
