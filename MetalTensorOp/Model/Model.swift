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
