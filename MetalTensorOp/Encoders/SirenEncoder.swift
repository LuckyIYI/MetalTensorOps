import Metal
import Foundation
import QuartzCore

enum SirenEncoderError: Error {
    case failedToLocateModelJson(String)
    case noMLPFoundInModelFile
    case failedToCreateBuffer
}

struct MLPTensorArguments {
    var weight: InlineArray<16, MTLResourceID>
    var bias: InlineArray<16, MTLResourceID>
    
    init() {
        weight = .init(repeating: .init())
        bias = .init(repeating: .init())
    }
}

final class SirenEncoder: ComputeEncoder {
    let pipelineState: MTLComputePipelineState
    var argumentTable: any MTL4ArgumentTable
    let residencySet: MTLResidencySet
    private let mlp: MLP
    private var timeBuffer: MTLBuffer
    private let tensorArgumentsBuffer: MTLBuffer

    init(device: MTLDevice, library: MTLLibrary, compiler: MTL4Compiler, queue: MTL4CommandQueue) throws {
        let functionDescriptor = MTL4LibraryFunctionDescriptor()
        functionDescriptor.name = "sirenMLP"
        functionDescriptor.library = library
        
        let pipelineDescriptor = MTL4ComputePipelineDescriptor()
        pipelineDescriptor.computeFunctionDescriptor = functionDescriptor
        
        self.pipelineState = try compiler.makeComputePipelineState(descriptor: pipelineDescriptor)
        
        self.pipelineState.reflection?.bindings.forEach { print($0) }
        
        let tableDesc = MTL4ArgumentTableDescriptor()
        tableDesc.maxTextureBindCount = 1
        tableDesc.maxBufferBindCount = 3
        
        let fileName = "siren"
        
        guard let url = Bundle.main.url(forResource: fileName, withExtension: "json") else {
            throw SirenEncoderError.failedToLocateModelJson(fileName)
        }
        let data = try Data(contentsOf: url)
        
        let sirenModel = try JSONDecoder().decode(SirenModel.self, from: data)
        guard let mlp = sirenModel.mlp else {
            throw SirenEncoderError.noMLPFoundInModelFile
        }
        self.mlp = mlp
        
        self.argumentTable = try device.makeArgumentTable(descriptor: tableDesc)

        let desc = MTL4ArgumentTableDescriptor()
        desc.maxBufferBindCount = 3
        desc.maxTextureBindCount = 1

        var mlpTensorArguments = MLPTensorArguments()
        for i in 0..<mlp.layers.count {
            mlpTensorArguments.weight[i] = mlp.layers[i].weightTensor.gpuResourceID
        }
        for i in 0..<mlp.layers.count {
            mlpTensorArguments.bias[i] = mlp.layers[i].biasTensor.gpuResourceID
        }

        guard let tensorArgumentsBuffer = device.makeBuffer(length: MemoryLayout<MLPTensorArguments>.stride, options: .storageModeShared) else {
            throw SirenEncoderError.failedToCreateBuffer
        }
        self.tensorArgumentsBuffer = tensorArgumentsBuffer
        memcpy(tensorArgumentsBuffer.contents(), &mlpTensorArguments, MemoryLayout<MLPTensorArguments>.stride)

        var layerCount32 = UInt32(mlp.layers.count)
        guard let layerCountBuffer = device.makeBuffer(length: MemoryLayout<UInt32>.size) else {
            throw SirenEncoderError.failedToCreateBuffer
        }
        layerCountBuffer.contents().copyMemory(from: &layerCount32, byteCount: MemoryLayout<UInt32>.size)

        guard let timeBuffer = device.makeBuffer(length: MemoryLayout<Float>.size, options: .storageModeShared) else {
            throw SirenEncoderError.failedToCreateBuffer
        }
        self.timeBuffer = timeBuffer

        self.argumentTable.setAddress(tensorArgumentsBuffer.gpuAddress, index: 0)
        self.argumentTable.setAddress(layerCountBuffer.gpuAddress, index: 1)
        self.argumentTable.setAddress(timeBuffer.gpuAddress, index: 2)

        let residency = try device.makeResidencySet(descriptor: .init())
        queue.addResidencySet(residency)

        for layer in mlp.layers {
            residency.addAllocation(layer.weightTensor)
            residency.addAllocation(layer.biasTensor)
        }

        residency.addAllocation(layerCountBuffer)
        residency.addAllocation(tensorArgumentsBuffer)

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
