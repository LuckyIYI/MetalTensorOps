import Foundation
import Metal
import simd

struct InstantNGPMetalWeights {
    let hashTable: MTLBuffer
    let mlp: MLP
    let imageWidth: Int
    let imageHeight: Int
}

struct InstantNGPSample {
    let position: SIMD2<Float>
    let value: SIMD3<Float>
}

struct InstantNGPWeightsFile: Codable {
    let model: ModelDescriptor?
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
        case model
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

        let hashTableCount = InstantNGPConfig.totalFeatures * (1 << InstantNGPConfig.log2HashmapSize)

        let hashFloats = encoding.hash_table.data
        guard !hashFloats.isEmpty else {
            throw InstantNGPError.invalidConfiguration
        }

        let floatBuffer = expand(values: hashFloats, expectedCount: hashTableCount)
        let hashData = floatBuffer.map { Float16($0) }
        guard let hashBuffer = device.makeBuffer(bytes: hashData, length: hashData.count * MemoryLayout<Float16>.stride, options: .storageModeShared) else {
            throw InstantNGPError.failedToCreateBuffer
        }

        let imageWidth = metadata.image?.width ?? 0
        let imageHeight = metadata.image?.height ?? 0

        return InstantNGPMetalWeights(
            hashTable: hashBuffer,
            mlp: mlp,
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

    func makeModelSampleDataset() throws -> [ModelSamplePoint] {
        return try makeSampleDataset().map { sample in
            ModelSamplePoint(position: sample.position, value: sample.value)
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
