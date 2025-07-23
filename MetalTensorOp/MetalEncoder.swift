import Metal
import Foundation

struct TensorArguments {
    var weight: StaticArray8<MTLResourceID>
    var bias:   StaticArray8<MTLResourceID>
    init() {
        weight = .init(repeating: .init())
        bias = .init(repeating: .init())
    }
}

class MetalEncoder {
    let pipelineState: MTLComputePipelineState
    var argumentTable: any MTL4ArgumentTable
    let residencySet: MTLResidencySet
    let mlp: MLP

    init(device: MTLDevice, library: MTLLibrary, compiler: MTL4Compiler, queue: MTL4CommandQueue) throws {
        let functionDescriptor = MTL4LibraryFunctionDescriptor()
        functionDescriptor.name = "mlp"
        functionDescriptor.library = library
        let pipelineDescriptor = MTL4ComputePipelineDescriptor()
        pipelineDescriptor.computeFunctionDescriptor = functionDescriptor
        self.pipelineState = try compiler.makeComputePipelineState(descriptor: pipelineDescriptor)
        
        self.pipelineState.reflection?.bindings.forEach { print($0) }
        let tableDesc = MTL4ArgumentTableDescriptor()
        tableDesc.maxTextureBindCount  = 1
        tableDesc.maxBufferBindCount = 2

        let url = Bundle.main.url(forResource: "weights", withExtension: "json")!
        let data = try Data(contentsOf: url)
        let mlp = try JSONDecoder().decode(MLP.self, from: data)
        
        self.mlp = mlp
        self.argumentTable = try device.makeArgumentTable(descriptor: tableDesc)


        let desc = MTL4ArgumentTableDescriptor()
        desc.maxBufferBindCount   = mlp.layers.count * 2    // tensors count as buffer slots
        desc.maxTextureBindCount  = 1

        var tensorArguements = TensorArguments()
        
        for i in 0..<mlp.layers.count {
            tensorArguements.weight[i] = mlp.layers[i].weights.gpuResourceID
        }
        for i in 0..<mlp.layers.count {
            tensorArguements.bias[i] = mlp.layers[i].biases.gpuResourceID
        }

        let tensorArgumentsBuffer = device.makeBuffer(length: MemoryLayout<TensorArguments>.stride, options: .storageModeShared)!
        memcpy(tensorArgumentsBuffer.contents(), &tensorArguements, MemoryLayout<TensorArguments>.stride)
        
        var layerCount32 = UInt32(mlp.layers.count)
        let layerCountBuffer = device.makeBuffer(length: MemoryLayout<UInt32>.size)!
        layerCountBuffer.contents().copyMemory(from: &layerCount32, byteCount: MemoryLayout<UInt32>.size)

        self.argumentTable.setAddress(tensorArgumentsBuffer.gpuAddress, index: 0)
        self.argumentTable.setAddress(layerCountBuffer.gpuAddress, index: 1)
        
        let residency = try device.makeResidencySet(descriptor: .init())
        queue.addResidencySet(residency)

        for l in mlp.layers {
            residency.addAllocation(l.rawWeights)
            residency.addAllocation(l.rawBiases)
            residency.addAllocation(l.weights)
            residency.addAllocation(l.biases)
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

