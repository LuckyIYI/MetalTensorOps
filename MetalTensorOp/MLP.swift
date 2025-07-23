import Foundation
import Metal

struct MLPParameterLayer: Codable {
    let weightTensor: MTLTensor
    let biasTensor: MTLTensor

    private enum CodingKeys: String, CodingKey {
        case weights
        case biases
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)

        let weightMatrix = try container.decode([[Float16]].self, forKey: .weights)
        let biasVector = try container.decode([Float16].self,  forKey: .biases)

        guard let device = MTLCreateSystemDefaultDevice() else {
            throw DecodingError.dataCorruptedError(
                forKey: .weights,
                in: container,
                debugDescription: "Metal device unavailable")
        }

        let rows = weightMatrix.count
        let cols = weightMatrix.first?.count ?? 0

        guard let wExtents = MTLTensorExtents([rows, cols]) else {
            throw DecodingError.dataCorruptedError(
                forKey: .weights,
                in: container,
                debugDescription: "Invalid weight extents")
        }
        let wDesc = MTLTensorDescriptor()
        wDesc.dimensions = wExtents
        wDesc.usage = .compute
        wDesc.strides  = .init([1, rows])!
        wDesc.dataType = .float16

        var flatW = [Float16]()
        flatW.reserveCapacity(rows * cols)
        for c in 0..<cols {
            for r in 0..<rows {
                flatW.append(Float16(weightMatrix[r][c]))
            }
        }
        let wBuffer = device.makeBuffer(bytes: flatW, length: flatW.count * MemoryLayout<Float16>.stride)!

        let wTensor = try wBuffer.makeTensor(descriptor: wDesc, offset: 0)

        guard let bExtents = MTLTensorExtents([biasVector.count]) else {
            throw DecodingError.dataCorruptedError(
                forKey: .biases,
                in: container,
                debugDescription: "Invalid bias extents")
        }
        let bDesc = MTLTensorDescriptor()
        bDesc.usage = .compute
        bDesc.dimensions = bExtents
        bDesc.strides = .init([1])!
        bDesc.dataType = .float16

        let flatB = biasVector.map { Float16($0) }
        let bBuffer = device.makeBuffer(bytes: flatB, length: flatB.count * MemoryLayout<Float16>.stride)!
        let bTensor = try bBuffer.makeTensor(descriptor: bDesc, offset: 0)

        self.weightTensor = wTensor
        self.biasTensor  = bTensor
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)

        let dims = weightTensor.dimensions.extents
        let rows = dims[0]
        let cols = dims[1]
        let dstStrides = MTLTensorExtents([cols, 1])!
        let count = rows * cols
        var flatW = [Float16](repeating: 0, count: count)
        weightTensor.getBytes(
            &flatW,
            strides: dstStrides,
            sliceOrigin: MTLTensorExtents([0, 0])!,
            sliceDimensions: weightTensor.dimensions)
        
        var matrix: [[Float16]] = []
        matrix.reserveCapacity(rows)
        for r in 0..<rows {
            let start = r * cols
            matrix.append(Array(flatW[start ..< start + cols]))
        }
        try container.encode(matrix, forKey: .weights)

        let biasCount = biasTensor.dimensions.extents[0]
        var biasArray = [Float16](repeating: 0, count: biasCount)
        biasTensor.getBytes(
            &biasArray,
            strides: biasTensor.strides,
            sliceOrigin: MTLTensorExtents([0])!,
            sliceDimensions: biasTensor.dimensions)

        try container.encode(biasArray, forKey: .biases)
    }
}

struct MLP: Codable {
    var layers: [MLPParameterLayer]

    private enum CodingKeys: String, CodingKey {
        case layers
    }

    init(layers: [MLPParameterLayer]) {
        self.layers = layers
    }

    init(from decoder: Decoder) throws {
        if let keyed = try? decoder.container(keyedBy: CodingKeys.self),
           keyed.contains(.layers) {
            self.layers = try keyed.decode([MLPParameterLayer].self, forKey: .layers)
        } else {
            let single = try decoder.singleValueContainer()
            self.layers = try single.decode([MLPParameterLayer].self)
        }
    }

    func encode(to encoder: Encoder) throws {
        var keyed = encoder.container(keyedBy: CodingKeys.self)
        try keyed.encode(layers, forKey: .layers)
    }
}
