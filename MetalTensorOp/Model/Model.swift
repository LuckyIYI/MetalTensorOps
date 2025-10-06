import Foundation
import Metal
import simd

enum ModelSampleError: Error {
    case missingSamples
    case invalidSampleDimensions
}

struct ModelSamplePoint {
    let position: SIMD2<Float>
    let value: SIMD3<Float>
}

struct ModelDescriptor: Codable {
    let type: String?
}

extension SirenModel {
    func makeSampleDataset() throws -> [ModelSamplePoint] {
        guard let samples else {
            throw ModelSampleError.missingSamples
        }

        return try samples.map { sample in
            guard sample.position.count == 2, sample.value.count == 3 else {
                throw ModelSampleError.invalidSampleDimensions
            }
            return ModelSamplePoint(
                position: SIMD2(Float(sample.position[0]), Float(sample.position[1])),
                value: SIMD3(Float(sample.value[0]), Float(sample.value[1]), Float(sample.value[2]))
            )
        }
    }
}

extension FourierModel {
    func makeSampleDataset() throws -> [ModelSamplePoint] {
        guard let samples else {
            throw ModelSampleError.missingSamples
        }
        return try samples.map { sample in
            guard sample.position.count == 2, sample.value.count == 3 else {
                throw ModelSampleError.invalidSampleDimensions
            }
            return ModelSamplePoint(
                position: SIMD2(Float(sample.position[0]), Float(sample.position[1])),
                value: SIMD3(Float(sample.value[0]), Float(sample.value[1]), Float(sample.value[2]))
            )
        }
    }
}

struct ModelSample: Codable {
    let position: [Float]
    let value: [Float]
}

struct SirenModel: Codable {
    let metadata: Metadata?
    let model: ModelDescriptor?
    let mlp: MLP?
    let sampleCount: Int?
    let sampleSeed: UInt64?
    let samples: [ModelSample]?

    private enum CodingKeys: String, CodingKey {
        case metadata
        case model
        case mlp
        case sampleCount = "sample_count"
        case sampleSeed = "sample_seed"
        case samples
    }

    init(metadata: Metadata?, model: ModelDescriptor?, mlp: MLP?, sampleCount: Int?, sampleSeed: UInt64?, samples: [ModelSample]?) {
        self.metadata = metadata
        self.model = model
        self.mlp = mlp
        self.sampleCount = sampleCount
        self.sampleSeed = sampleSeed
        self.samples = samples
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.metadata = try container.decodeIfPresent(Metadata.self, forKey: .metadata)
        self.model = try container.decodeIfPresent(ModelDescriptor.self, forKey: .model)
        self.mlp = try container.decodeIfPresent(MLP.self, forKey: .mlp)
        self.sampleCount = try container.decodeIfPresent(Int.self, forKey: .sampleCount)
        self.sampleSeed = try container.decodeIfPresent(UInt64.self, forKey: .sampleSeed)
        self.samples = try container.decodeIfPresent([ModelSample].self, forKey: .samples)
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encodeIfPresent(metadata, forKey: .metadata)
        try container.encodeIfPresent(model, forKey: .model)
        try container.encodeIfPresent(mlp, forKey: .mlp)
        try container.encodeIfPresent(sampleCount, forKey: .sampleCount)
        try container.encodeIfPresent(sampleSeed, forKey: .sampleSeed)
        try container.encodeIfPresent(samples, forKey: .samples)
    }
}

struct FourierModel: Codable {
    let metadata: Metadata?
    let model: ModelDescriptor?
    let mlp: MLP?
    let fourier: FourierParams
    let sampleCount: Int?
    let sampleSeed: UInt64?
    let samples: [ModelSample]?

    private enum CodingKeys: String, CodingKey {
        case metadata
        case model
        case mlp
        case fourier
        case sampleCount = "sample_count"
        case sampleSeed = "sample_seed"
        case samples
    }

    init(metadata: Metadata?, model: ModelDescriptor?, mlp: MLP?, fourier: FourierParams, sampleCount: Int?, sampleSeed: UInt64?, samples: [ModelSample]?) {
        self.metadata = metadata
        self.model = model
        self.mlp = mlp
        self.fourier = fourier
        self.sampleCount = sampleCount
        self.sampleSeed = sampleSeed
        self.samples = samples
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.metadata = try container.decodeIfPresent(Metadata.self, forKey: .metadata)
        self.model = try container.decodeIfPresent(ModelDescriptor.self, forKey: .model)
        self.mlp = try container.decodeIfPresent(MLP.self, forKey: .mlp)
        self.fourier = try container.decode(FourierParams.self, forKey: .fourier)
        self.sampleCount = try container.decodeIfPresent(Int.self, forKey: .sampleCount)
        self.sampleSeed = try container.decodeIfPresent(UInt64.self, forKey: .sampleSeed)
        self.samples = try container.decodeIfPresent([ModelSample].self, forKey: .samples)
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encodeIfPresent(metadata, forKey: .metadata)
        try container.encodeIfPresent(model, forKey: .model)
        try container.encodeIfPresent(mlp, forKey: .mlp)
        try container.encode(fourier, forKey: .fourier)
        try container.encodeIfPresent(sampleCount, forKey: .sampleCount)
        try container.encodeIfPresent(sampleSeed, forKey: .sampleSeed)
        try container.encodeIfPresent(samples, forKey: .samples)
    }
}

struct Metadata: Codable {
    let mode: String

    struct ImageMetadata: Codable {
        let width: Int?
        let height: Int?
        let aspect_ratio: Float?
    }

    struct SDFMetadata: Codable {
        let resolution: Int?
    }

    let image: ImageMetadata?
    let sdf: SDFMetadata?
}

struct FourierParams: Codable {
    let bTensor: MTLTensor
    let sigma: Float?

    private enum CodingKeys: String, CodingKey {
        case B
        case sigma
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)

        let bMatrix = try container.decode([[Float]].self, forKey: .B)
        self.sigma = try container.decodeIfPresent(Float.self, forKey: .sigma)

        guard let device = MTLCreateSystemDefaultDevice() else {
            throw DecodingError.dataCorruptedError(
                forKey: .B,
                in: container,
                debugDescription: "Metal device unavailable")
        }

        let rows = bMatrix.count
        let cols = bMatrix.first?.count ?? 0

        guard let bExtents = MTLTensorExtents([rows, cols]) else {
            throw DecodingError.dataCorruptedError(
                forKey: .B,
                in: container,
                debugDescription: "Invalid B matrix extents")
        }
        let bDesc = MTLTensorDescriptor()
        bDesc.dimensions = bExtents
        bDesc.usage = .compute
        bDesc.strides  = .init([1, rows])!
        bDesc.dataType = .float32

        var flatB = [Float]()
        flatB.reserveCapacity(rows * cols)
        
        // Transpose from row-major (list of lists) to column-major for the buffer
        for c in 0..<cols {
            for r in 0..<rows {
                flatB.append(bMatrix[r][c])
            }
        }
        let bBuffer = device.makeBuffer(bytes: flatB, length: flatB.count * MemoryLayout<Float>.stride)!

        self.bTensor = try bBuffer.makeTensor(descriptor: bDesc, offset: 0)
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encodeIfPresent(sigma, forKey: .sigma)

        let dims = bTensor.dimensions.extents
        let rows = dims[0]
        let cols = dims[1]
        let count = rows * cols
        var flatB = [Float](repeating: 0, count: count)

        let dstStrides = MTLTensorExtents([cols, 1])!
        bTensor.getBytes(
            &flatB,
            strides: dstStrides,
            sliceOrigin: MTLTensorExtents([0, 0])!,
            sliceDimensions: bTensor.dimensions)

        var matrix: [[Float]] = []
        matrix.reserveCapacity(rows)
        for r in 0..<rows {
            let start = r * cols
            matrix.append(Array(flatB[start ..< start + cols]))
        }
        try container.encode(matrix, forKey: .B)
    }
}

struct InstantNGPMetalWeights {
    let hashTable: MTLBuffer
    let encoding: InstantNGPEncoding
    let mlp: MLP
    let imageWidth: Int
    let imageHeight: Int
}

struct InstantNGPEncoding: Codable {
    struct HashTable: Codable {
        let shape: [Int]
        let data: [Float]
    }

    let numLevels: Int
    let featuresPerLevel: Int
    let log2HashmapSize: Int
    let baseResolution: Int
    let maxResolution: Int
    let hashTable: HashTable

    private enum CodingKeys: String, CodingKey {
        case numLevels = "num_levels"
        case featuresPerLevel = "features_per_level"
        case log2HashmapSize = "log2_hashmap_size"
        case baseResolution = "base_resolution"
        case maxResolution = "max_resolution"
        case hashTable = "hash_table"
    }
}

struct InstantNGPModel: Codable {
    let metadata: Metadata?
    let model: ModelDescriptor?
    let encoding: InstantNGPEncoding
    let mlp: MLP?
    let sampleCount: Int?
    let sampleSeed: UInt64?
    let samples: [ModelSample]?

    private enum CodingKeys: String, CodingKey {
        case metadata
        case model
        case encoding
        case mlp
        case sampleCount = "sample_count"
        case sampleSeed = "sample_seed"
        case samples
    }
}

extension InstantNGPModel {
    static func load(from url: URL) throws -> InstantNGPModel {
        let data = try Data(contentsOf: url)
        return try JSONDecoder().decode(InstantNGPModel.self, from: data)
    }

    func makeMetalWeights(device: MTLDevice) throws -> InstantNGPMetalWeights {
        guard let mlp else {
            throw InstantNGPError.invalidConfiguration
        }
        guard mlp.layers.count >= 2 else {
            throw InstantNGPError.invalidConfiguration
        }

        guard encoding.numLevels == InstantNGPConfig.numLevels,
              encoding.featuresPerLevel == InstantNGPConfig.featuresPerLevel,
              encoding.log2HashmapSize == InstantNGPConfig.log2HashmapSize,
              encoding.baseResolution == InstantNGPConfig.baseResolution,
              encoding.maxResolution == InstantNGPConfig.maxResolution else {
            throw InstantNGPError.invalidConfiguration
        }

        let expectedHashCount = encoding.numLevels * encoding.featuresPerLevel * (1 << encoding.log2HashmapSize)
        let expandedHash = expandHashData(values: encoding.hashTable.data, expectedCount: expectedHashCount)

        let hashData = expandedHash.map { Float16($0) }
        guard let hashBuffer = device.makeBuffer(
            bytes: hashData,
            length: hashData.count * MemoryLayout<Float16>.stride,
            options: .storageModeShared
        ) else {
            throw InstantNGPError.failedToCreateBuffer
        }

        let imageWidth = metadata?.image?.width ?? 0
        let imageHeight = metadata?.image?.height ?? 0

        return InstantNGPMetalWeights(
            hashTable: hashBuffer,
            encoding: encoding,
            mlp: mlp,
            imageWidth: imageWidth,
            imageHeight: imageHeight
        )
    }

    func makeSampleDataset() throws -> [ModelSamplePoint] {
        guard let samples else {
            throw ModelSampleError.missingSamples
        }

        return try samples.map { sample in
            guard sample.position.count == 2, sample.value.count == 3 else {
                throw ModelSampleError.invalidSampleDimensions
            }
            return ModelSamplePoint(
                position: SIMD2(Float(sample.position[0]), Float(sample.position[1])),
                value: SIMD3(Float(sample.value[0]), Float(sample.value[1]), Float(sample.value[2]))
            )
        }
    }

    func makeModelSampleDataset() throws -> [ModelSamplePoint] {
        try makeSampleDataset()
    }

    private func expandHashData(values: [Float], expectedCount: Int) -> [Float] {
        guard !values.isEmpty else {
            return Array(repeating: 0, count: expectedCount)
        }

        if values.count == expectedCount {
            return values
        }

        var expanded = [Float](repeating: 0, count: expectedCount)
        for index in 0..<expectedCount {
            expanded[index] = values[index % values.count]
        }
        return expanded
    }
}
