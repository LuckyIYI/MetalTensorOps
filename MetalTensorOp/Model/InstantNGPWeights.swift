import Foundation
import Metal
import simd

struct InstantNGPMetalWeights {
    let hashTable: MTLBuffer
    let layer1Weights: MTLTensor
    let layer1Bias: MTLTensor
    let layer2Weights: MTLTensor
    let layer2Bias: MTLTensor
    let imageWidth: Int
    let imageHeight: Int
}

struct InstantNGPSample {
    let position: SIMD2<Float>
    let value: SIMD3<Float>
}

struct InstantNGPWeightsFile: Codable {
    struct Metadata: Codable {
        struct ImageMetadata: Codable {
            let width: Int?
            let height: Int?
            let aspect_ratio: Float?
        }

        let mode: String?
        let image: ImageMetadata?
    }

    struct Encoding: Codable {
        struct HashTable: Codable {
            let shape: [Int]
            let data: [Float]
        }

        let num_levels: Int
        let features_per_level: Int
        let log2_hashmap_size: Int
        let base_resolution: Int
        let max_resolution: Int
        let hash_table: HashTable
    }

    struct MLP: Codable {
        struct Layer: Codable {
            let weights: [Float]
            let biases: [Float]
        }

        let hidden_width: Int
        let output_dim: Int
        let layers: [Layer]
    }

    struct Sample: Codable {
        let position: [Float]
        let value: [Float]
    }

    let metadata: Metadata
    let encoding: Encoding
    let mlp: MLP
    let samples: [Sample]?
    let sampleCount: Int?
    let sampleSeed: UInt64?

    enum CodingKeys: String, CodingKey {
        case metadata
        case encoding
        case mlp
        case samples
        case sampleCount = "sample_count"
        case sampleSeed = "sample_seed"
    }
}

extension InstantNGPWeightsFile {
    static func load(from url: URL) throws -> InstantNGPWeightsFile {
        let data = try Data(contentsOf: url)
        return try JSONDecoder().decode(InstantNGPWeightsFile.self, from: data)
    }

    func makeMetalWeights(device: MTLDevice) throws -> InstantNGPMetalWeights {
        guard mlp.layers.count >= 2 else {
            throw InstantNGPError.invalidConfiguration
        }

        let totalFeatures = InstantNGPConfig.totalFeatures
        let hiddenWidth = InstantNGPConfig.mlpHiddenWidth
        let outputDim = InstantNGPConfig.mlpOutputDim
        let hashTableCount = InstantNGPConfig.totalFeatures * (1 << InstantNGPConfig.log2HashmapSize)

        // Hash table buffer (float16)
        let hashFloats = encoding.hash_table.data
        guard !hashFloats.isEmpty else {
            throw InstantNGPError.invalidConfiguration
        }

        let floatBuffer = expand(values: hashFloats, expectedCount: hashTableCount)
        let hashData = floatBuffer.map { Float16($0) }
        guard let hashBuffer = device.makeBuffer(bytes: hashData, length: hashData.count * MemoryLayout<Float16>.stride, options: .storageModeShared) else {
            throw InstantNGPError.failedToCreateBuffer
        }

        // Layer 1 weights tensor
        let l1Expected = totalFeatures * hiddenWidth
        let layer1WeightsF16 = expand(values: mlp.layers[0].weights, expectedCount: l1Expected).map { Float16($0) }
        guard let layer1WeightsBuffer = device.makeBuffer(bytes: layer1WeightsF16, length: layer1WeightsF16.count * MemoryLayout<Float16>.stride, options: .storageModeShared) else {
            throw InstantNGPError.failedToCreateBuffer
        }
        let layer1WeightsDesc = MTLTensorDescriptor()
        layer1WeightsDesc.dimensions = MTLTensorExtents([hiddenWidth, totalFeatures])!
        layer1WeightsDesc.strides = MTLTensorExtents([1, hiddenWidth])!
        layer1WeightsDesc.usage = .compute
        layer1WeightsDesc.dataType = .float16
        let layer1WeightsTensor = try layer1WeightsBuffer.makeTensor(descriptor: layer1WeightsDesc, offset: 0)

        // Layer 1 bias tensor
        let layer1BiasF16 = expand(values: mlp.layers[0].biases, expectedCount: hiddenWidth).map { Float16($0) }
        guard let layer1BiasBuffer = device.makeBuffer(bytes: layer1BiasF16, length: layer1BiasF16.count * MemoryLayout<Float16>.stride, options: .storageModeShared) else {
            throw InstantNGPError.failedToCreateBuffer
        }
        let layer1BiasDesc = MTLTensorDescriptor()
        layer1BiasDesc.dimensions = MTLTensorExtents([hiddenWidth])!
        layer1BiasDesc.strides = MTLTensorExtents([1])!
        layer1BiasDesc.usage = .compute
        layer1BiasDesc.dataType = .float16
        let layer1BiasTensor = try layer1BiasBuffer.makeTensor(descriptor: layer1BiasDesc, offset: 0)

        // Layer 2 weights tensor
        let l2Expected = hiddenWidth * outputDim
        let layer2WeightsF16 = expand(values: mlp.layers[1].weights, expectedCount: l2Expected).map { Float16($0) }
        guard let layer2WeightsBuffer = device.makeBuffer(bytes: layer2WeightsF16, length: layer2WeightsF16.count * MemoryLayout<Float16>.stride, options: .storageModeShared) else {
            throw InstantNGPError.failedToCreateBuffer
        }
        let layer2WeightsDesc = MTLTensorDescriptor()
        layer2WeightsDesc.dimensions = MTLTensorExtents([outputDim, hiddenWidth])!
        layer2WeightsDesc.strides = MTLTensorExtents([1, outputDim])!
        layer2WeightsDesc.usage = .compute
        layer2WeightsDesc.dataType = .float16
        let layer2WeightsTensor = try layer2WeightsBuffer.makeTensor(descriptor: layer2WeightsDesc, offset: 0)

        // Layer 2 bias tensor
        let layer2BiasF16 = expand(values: mlp.layers[1].biases, expectedCount: outputDim).map { Float16($0) }
        guard let layer2BiasBuffer = device.makeBuffer(bytes: layer2BiasF16, length: layer2BiasF16.count * MemoryLayout<Float16>.stride, options: .storageModeShared) else {
            throw InstantNGPError.failedToCreateBuffer
        }
        let layer2BiasDesc = MTLTensorDescriptor()
        layer2BiasDesc.dimensions = MTLTensorExtents([outputDim])!
        layer2BiasDesc.strides = MTLTensorExtents([1])!
        layer2BiasDesc.usage = .compute
        layer2BiasDesc.dataType = .float16
        let layer2BiasTensor = try layer2BiasBuffer.makeTensor(descriptor: layer2BiasDesc, offset: 0)

        let imageWidth = metadata.image?.width ?? 0
        let imageHeight = metadata.image?.height ?? 0

        return InstantNGPMetalWeights(
            hashTable: hashBuffer,
            layer1Weights: layer1WeightsTensor,
            layer1Bias: layer1BiasTensor,
            layer2Weights: layer2WeightsTensor,
            layer2Bias: layer2BiasTensor,
            imageWidth: imageWidth,
            imageHeight: imageHeight
        )
    }

    func makeSampleDataset() throws -> [InstantNGPSample] {
        guard let samples = samples else {
            throw InstantNGPError.invalidConfiguration
        }

        return try samples.map { entry in
            guard entry.position.count == 2, entry.value.count == 3 else {
                throw InstantNGPError.invalidConfiguration
            }
            let position = SIMD2<Float>(entry.position[0], entry.position[1])
            let value = SIMD3<Float>(entry.value[0], entry.value[1], entry.value[2])
            return InstantNGPSample(position: position, value: value)
        }
    }

    private func expand(values: [Float], expectedCount: Int) -> [Float] {
        guard !values.isEmpty else {
            return Array(repeating: 0, count: expectedCount)
        }
        if values.count == expectedCount {
            return values
        }
        return (0..<expectedCount).map { values[$0 % values.count] }
    }
}
