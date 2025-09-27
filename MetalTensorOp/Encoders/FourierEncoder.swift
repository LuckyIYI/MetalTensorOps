import Metal
import Foundation
import QuartzCore

enum FourierEncoderError: Error {
    case failedToLocateModelJson
    case noMLPFoundInModelFile
    case fourierParamsMissing
    case failedToCreateBuffer
}

struct FourierTensorArguments {
    var weight: InlineArray<16, MTLResourceID>
    var bias: InlineArray<16, MTLResourceID>
    var bMatrix: MTLResourceID

    init() {
        weight = .init(repeating: .init())
        bias = .init(repeating: .init())
        bMatrix = .init()
    }
}

final class FourierEncoder: ComputeEncoder {
    let pipelineState: MTLComputePipelineState
    var argumentTable: any MTL4ArgumentTable
    let residencySet: MTLResidencySet
    private let mlp: MLP
    private let fourier: FourierParams
    private var timeBuffer: MTLBuffer
    private let tensorArgumentsBuffer: MTLBuffer
    private let layerCountBuffer: MTLBuffer
    private let sigmaBuffer: MTLBuffer

    init(device: MTLDevice, library: MTLLibrary, compiler: MTL4Compiler, queue: MTL4CommandQueue) throws {
        let functionDescriptor = MTL4LibraryFunctionDescriptor()
        functionDescriptor.name = "fourierMLP"
        functionDescriptor.library = library
        let pipelineDescriptor = MTL4ComputePipelineDescriptor()
        pipelineDescriptor.computeFunctionDescriptor = functionDescriptor
        self.pipelineState = try compiler.makeComputePipelineState(descriptor: pipelineDescriptor)
        self.pipelineState.reflection?.bindings.forEach { print($0) }
        let tableDesc = MTL4ArgumentTableDescriptor()
        tableDesc.maxTextureBindCount = 1
        tableDesc.maxBufferBindCount = 5

        guard let url = Bundle.main.url(forResource: "fourier", withExtension: "json") else {
            throw FourierEncoderError.failedToLocateModelJson
        }
        let data = try Data(contentsOf: url)
        let fourierModel = try JSONDecoder().decode(FourierModel.self, from: data)
        guard let mlp = fourierModel.mlp else {
            throw FourierEncoderError.noMLPFoundInModelFile
        }
        let fourier = fourierModel.fourier
        self.mlp = mlp
        self.fourier = fourier
        self.argumentTable = try device.makeArgumentTable(descriptor: tableDesc)

        var mlpTensorArguments = FourierTensorArguments()
        for i in 0..<mlp.layers.count {
            mlpTensorArguments.weight[i] = mlp.layers[i].weightTensor.gpuResourceID
        }
        for i in 0..<mlp.layers.count {
            mlpTensorArguments.bias[i] = mlp.layers[i].biasTensor.gpuResourceID
        }
        guard let sigma = fourier.sigma else {
            throw FourierEncoderError.fourierParamsMissing
        }

        mlpTensorArguments.bMatrix = fourier.bTensor.gpuResourceID

        guard let tensorArgumentsBuffer = device.makeBuffer(length: MemoryLayout<FourierTensorArguments>.stride, options: .storageModeShared) else {
            throw FourierEncoderError.failedToCreateBuffer
        }
        self.tensorArgumentsBuffer = tensorArgumentsBuffer
        memcpy(tensorArgumentsBuffer.contents(), &mlpTensorArguments, MemoryLayout<FourierTensorArguments>.stride)

        var layerCount32 = UInt32(mlp.layers.count)
        guard let layerCountBuffer = device.makeBuffer(length: MemoryLayout<UInt32>.size, options: .storageModeShared) else {
            throw FourierEncoderError.failedToCreateBuffer
        }
        layerCountBuffer.contents().copyMemory(from: &layerCount32, byteCount: MemoryLayout<UInt32>.size)
        self.layerCountBuffer = layerCountBuffer

        guard let timeBuffer = device.makeBuffer(length: MemoryLayout<Float>.size, options: .storageModeShared) else {
            throw FourierEncoderError.failedToCreateBuffer
        }
        self.timeBuffer = timeBuffer
        
        guard let sigmaBuffer = device.makeBuffer(length: MemoryLayout<Float>.stride, options: .storageModeShared) else {
            throw FourierEncoderError.failedToCreateBuffer
        }
        var sigmaCopy = sigma
        memcpy(sigmaBuffer.contents(), &sigmaCopy, MemoryLayout<Float>.stride)
        self.sigmaBuffer = sigmaBuffer

        self.argumentTable.setAddress(tensorArgumentsBuffer.gpuAddress, index: 0)
        self.argumentTable.setAddress(layerCountBuffer.gpuAddress, index: 1)
        self.argumentTable.setAddress(sigmaBuffer.gpuAddress, index: 2)
        self.argumentTable.setAddress(timeBuffer.gpuAddress, index: 3)

        let residency = try device.makeResidencySet(descriptor: .init())
        queue.addResidencySet(residency)
        for layer in mlp.layers {
            residency.addAllocation(layer.weightTensor)
            residency.addAllocation(layer.biasTensor)
            
            if let weightBuffer = layer.weightTensor as? MTLBuffer {
                residency.addAllocation(weightBuffer)
            }
            
            if let biasBuffer = layer.biasTensor as? MTLBuffer {
                residency.addAllocation(biasBuffer)
            }
        }

        residency.addAllocation(layerCountBuffer)
        residency.addAllocation(fourier.bTensor)
        if let bBuffer = fourier.bTensor as? MTLBuffer {
            residency.addAllocation(bBuffer)
        }
        residency.addAllocation(tensorArgumentsBuffer)
        residency.addAllocation(timeBuffer)
        residency.addAllocation(sigmaBuffer)

        residency.commit()
        self.residencySet = residency
    }

    func encode(drawableTexture: MTLTexture, commandBuffer: MTL4CommandBuffer) {
        var t = Float(CACurrentMediaTime())
        memcpy(timeBuffer.contents(), &t, MemoryLayout<Float>.size)

        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        commandBuffer.useResidencySet(residencySet)
        encoder.setComputePipelineState(pipelineState)
        encoder.setArgumentTable(argumentTable)
        argumentTable.setTexture(drawableTexture.gpuResourceID, index: 0)
        encoder.dispatchThreads(
            threadsPerGrid: .init(width: drawableTexture.width, height: drawableTexture.height, depth: 1),
            threadsPerThreadgroup: .init(width: 32, height: 32, depth: 1)
        )
        encoder.endEncoding()
    }
}
