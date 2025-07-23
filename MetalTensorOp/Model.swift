import Foundation
import Metal


struct SirenModel: Codable {
    let metadata: Metadata?
    let mlp: MLP?

    private enum CodingKeys: String, CodingKey {
        case metadata
        case mlp
    }

    init(metadata: Metadata?, mlp: MLP?) {
        self.metadata = metadata
        self.mlp = mlp
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.metadata = try container.decodeIfPresent(Metadata.self, forKey: .metadata)
        self.mlp = try container.decodeIfPresent(MLP.self, forKey: .mlp)
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encodeIfPresent(metadata, forKey: .metadata)
        try container.encodeIfPresent(mlp, forKey: .mlp)
    }
}

struct FourierModel: Codable {
    let metadata: Metadata?
    let mlp: MLP?
    let fourier: FourierParams

    private enum CodingKeys: String, CodingKey {
        case metadata
        case mlp
        case fourier
    }

    init(metadata: Metadata?, mlp: MLP?, fourier: FourierParams) {
        self.metadata = metadata
        self.mlp = mlp
        self.fourier = fourier
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.metadata = try container.decodeIfPresent(Metadata.self, forKey: .metadata)
        self.mlp = try container.decodeIfPresent(MLP.self, forKey: .mlp)
        self.fourier = try container.decode(FourierParams.self, forKey: .fourier)
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encodeIfPresent(metadata, forKey: .metadata)
        try container.encodeIfPresent(mlp, forKey: .mlp)
        try container.encode(fourier, forKey: .fourier)
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
