import Foundation
import Metal


struct Layer: Codable {
    let weights: MTLTensor
    let biases: MTLTensor
    
    let rawWeights: MTLBuffer
    let rawBiases: MTLBuffer

    private enum CodingKeys: String, CodingKey {
        case weight
        case bias
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)

        let weightMatrix = try container.decode([[Float16]].self, forKey: .weight)
        let biasVector = try container.decode([Float16].self,  forKey: .bias)

        guard let device = MTLCreateSystemDefaultDevice() else {
            throw DecodingError.dataCorruptedError(
                forKey: .weight,
                in: container,
                debugDescription: "Metal device unavailable")
        }

        let rows = weightMatrix.count
        let cols = weightMatrix.first?.count ?? 0

        // ---- Weights --------------------------------------------------------
        guard let wExtents = MTLTensorExtents([rows, cols]) else {
            throw DecodingError.dataCorruptedError(
                forKey: .weight,
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


        // ---- Biases ---------------------------------------------------------
        guard let bExtents = MTLTensorExtents([biasVector.count]) else {
            throw DecodingError.dataCorruptedError(
                forKey: .bias,
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

        self.weights = wTensor
        self.biases  = bTensor
        self.rawWeights = wBuffer
        self.rawBiases  = bBuffer
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)

        // ---- Weights --------------------------------------------------------
        let dims = weights.dimensions.extents
        let rows = dims[0]
        let cols = dims[1]
        let dstStrides = MTLTensorExtents([cols, 1])!
        let count = rows * cols
        var flatW = [Float16](repeating: 0, count: count)
        weights.getBytes(
            &flatW,
            strides: dstStrides,
            sliceOrigin: MTLTensorExtents([0, 0])!,
            sliceDimensions: weights.dimensions)
        
        var matrix: [[Float16]] = []
        matrix.reserveCapacity(rows)
        for r in 0..<rows {
            let start = r * cols
            matrix.append(Array(flatW[start ..< start + cols]))
        }
        try container.encode(matrix, forKey: .weight)

        // ---- Biases ---------------------------------------------------------
        let biasCount = biases.dimensions.extents[0]
        var biasArray = [Float16](repeating: 0, count: biasCount)
        biases.getBytes(
            &biasArray,
            strides: biases.strides,
            sliceOrigin: MTLTensorExtents([0])!,
            sliceDimensions: biases.dimensions)

        try container.encode(biasArray, forKey: .bias)
    }
}

/// A simple multi‑layer perceptron composed of several `Layer`s.
struct MLP: Codable {
    var layers: [Layer]

    private enum CodingKeys: String, CodingKey { case layers }

    init(layers: [Layer]) { self.layers = layers }

    init(from decoder: Decoder) throws {
        if let keyed = try? decoder.container(keyedBy: CodingKeys.self),
           keyed.contains(.layers) {
            self.layers = try keyed.decode([Layer].self, forKey: .layers)
            return
        }

        let single = try decoder.singleValueContainer()
        self.layers = try single.decode([Layer].self)
    }

    func encode(to encoder: Encoder) throws {
        var single = encoder.singleValueContainer()
        try single.encode(layers)
    }
}
