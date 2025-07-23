import Metal
import Foundation

struct FourierTensorArguments {
    var weight: StaticArray16<MTLResourceID>
    var bias: StaticArray16<MTLResourceID>
    var bMatrix: MTLResourceID
    // Add Fourier-specific arguments if desired
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
    let mlp: MLP
    let fourier: FourierParams

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
        tableDesc.maxBufferBindCount = 4

        guard let url = Bundle.main.url(forResource: "fourier", withExtension: "json") else {
            throw NSError(domain: "FourierEncoder", code: -1, userInfo: [NSLocalizedDescriptionKey : "Failed to locate model.json"])
        }
        let data = try Data(contentsOf: url)
        let fourierModel = try JSONDecoder().decode(FourierModel.self, from: data)
        guard let mlp = fourierModel.mlp else {
            throw NSError(domain: "FourierEncoder", code: -1, userInfo: [NSLocalizedDescriptionKey : "No MLP found in Fourier model file"])
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
            throw NSError(domain: "FourierEncoder", code: -1, userInfo: [NSLocalizedDescriptionKey : "FourierParams missing sigma"])
        }

        mlpTensorArguments.bMatrix = fourier.bTensor.gpuResourceID

        guard let tensorArgumentsBuffer = device.makeBuffer(length: MemoryLayout<FourierTensorArguments>.stride, options: .storageModeShared) else {
            throw NSError(domain: "FourierEncoder", code: -1, userInfo: [NSLocalizedDescriptionKey : "Failed to create tensorArgumentsBuffer"])
        }
        memcpy(tensorArgumentsBuffer.contents(), &mlpTensorArguments, MemoryLayout<FourierTensorArguments>.stride)

        var layerCount32 = UInt32(mlp.layers.count)
        guard let layerCountBuffer = device.makeBuffer(length: MemoryLayout<UInt32>.size) else {
            throw NSError(domain: "FourierEncoder", code: -1, userInfo: [NSLocalizedDescriptionKey : "Failed to create layerCountBuffer"])
        }
        layerCountBuffer.contents().copyMemory(from: &layerCount32, byteCount: MemoryLayout<UInt32>.size)

        guard let sigmaBuffer = device.makeBuffer(length: MemoryLayout<Float>.stride, options: .storageModeShared) else {
            throw NSError(domain: "FourierEncoder", code: -1, userInfo: [NSLocalizedDescriptionKey : "Failed to create sigmaBuffer"])
        }
        var sigmaCopy = sigma
        memcpy(sigmaBuffer.contents(), &sigmaCopy, MemoryLayout<Float>.stride)

        guard let sigmaBuffer = device.makeBuffer(length: MemoryLayout<Float>.stride, options: .storageModeShared) else {
            throw NSError(domain: "FourierEncoder", code: -1, userInfo: [NSLocalizedDescriptionKey : "Failed to create sigmaBuffer"])
        }

        self.argumentTable.setAddress(tensorArgumentsBuffer.gpuAddress, index: 0)
        self.argumentTable.setAddress(layerCountBuffer.gpuAddress, index: 1)
        self.argumentTable.setAddress(sigmaBuffer.gpuAddress, index: 2)

        let residency = try device.makeResidencySet(descriptor: .init())
        queue.addResidencySet(residency)
        for layer in mlp.layers {
            residency.addAllocation(layer.weightTensor)
            residency.addAllocation(layer.biasTensor)
        }
        residency.addAllocation(layerCountBuffer)
        residency.addAllocation(fourier.bTensor)
        residency.addAllocation(sigmaBuffer)
        residency.addAllocation(sigmaBuffer)
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
