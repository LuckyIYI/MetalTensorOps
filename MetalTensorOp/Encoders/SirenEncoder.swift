// This file only implements SirenEncoder. FourierEncoder is implemented separately.

import Metal
import Foundation

struct MLPTensorArguments {
    var weight: StaticArray16<MTLResourceID>
    var bias: StaticArray16<MTLResourceID>
    
    init() {
        weight = .init(repeating: .init())
        bias = .init(repeating: .init())
    }
}

final class SirenEncoder: ComputeEncoder {
    let pipelineState: MTLComputePipelineState
    var argumentTable: any MTL4ArgumentTable
    let residencySet: MTLResidencySet
    let mlp: MLP


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
        tableDesc.maxBufferBindCount = 2
        
        let fileName = "siren"
        
        guard let url = Bundle.main.url(forResource: fileName, withExtension: "json") else {
            throw NSError(domain: "MetalEncoder", code: -1, userInfo: [NSLocalizedDescriptionKey : "Failed to locate \(fileName).json"])
        }
        let data = try Data(contentsOf: url)
        
        let sirenModel = try JSONDecoder().decode(SirenModel.self, from: data)
        guard let mlp = sirenModel.mlp else {
            throw NSError(domain: "MetalEncoder", code: -1, userInfo: [NSLocalizedDescriptionKey : "No MLP found in siren model file"])
        }
        self.mlp = mlp
        
        self.argumentTable = try device.makeArgumentTable(descriptor: tableDesc)

        let desc = MTL4ArgumentTableDescriptor()
        desc.maxBufferBindCount = 2
        desc.maxTextureBindCount = 1

        var mlpTensorArguments = MLPTensorArguments()

        for i in 0..<mlp.layers.count {
            mlpTensorArguments.weight[i] = mlp.layers[i].weightTensor.gpuResourceID
        }
        for i in 0..<mlp.layers.count {
            mlpTensorArguments.bias[i] = mlp.layers[i].biasTensor.gpuResourceID
        }

        guard let tensorArgumentsBuffer = device.makeBuffer(length: MemoryLayout<MLPTensorArguments>.stride, options: .storageModeShared) else {
            throw NSError(domain: "MetalEncoder", code: -1, userInfo: [NSLocalizedDescriptionKey : "Failed to create tensorArgumentsBuffer"])
        }
        memcpy(tensorArgumentsBuffer.contents(), &mlpTensorArguments, MemoryLayout<MLPTensorArguments>.stride)

        var layerCount32 = UInt32(mlp.layers.count)
        guard let layerCountBuffer = device.makeBuffer(length: MemoryLayout<UInt32>.size) else {
            throw NSError(domain: "MetalEncoder", code: -1, userInfo: [NSLocalizedDescriptionKey : "Failed to create layerCountBuffer"])
        }
        layerCountBuffer.contents().copyMemory(from: &layerCount32, byteCount: MemoryLayout<UInt32>.size)

        self.argumentTable.setAddress(tensorArgumentsBuffer.gpuAddress, index: 0)
        self.argumentTable.setAddress(layerCountBuffer.gpuAddress, index: 1)

        let residency = try device.makeResidencySet(descriptor: .init())
        queue.addResidencySet(residency)

        for layer in mlp.layers {
            residency.addAllocation(layer.weightTensor)
            residency.addAllocation(layer.biasTensor)
            residency.addAllocation(layerCountBuffer)
        }

        residency.commit()
        self.residencySet = residency
    }

    func encode(drawableTexture: MTLTexture, commandBuffer: MTL4CommandBuffer) {
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

