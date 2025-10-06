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
    var mlp: MetalMLPTensorArguments
    var bMatrix: MTLResourceID

    init() {
        mlp = .init()
        bMatrix = .init()
    }
}

final class FourierEncoder: ComputeEncoder {
    private struct RenderUniforms {
        var time: Float
        var trainingWidth: UInt32
        var trainingHeight: UInt32
        var padding: UInt32 = 0
    }

    let pipelineState: MTLComputePipelineState
    let cooperativePipelineState: MTLComputePipelineState
    let cooperativeBufferPipelineState: MTLComputePipelineState
    var argumentTable: any MTL4ArgumentTable
    var bufferArgumentTable: any MTL4ArgumentTable
    let residencySet: MTLResidencySet
    private let mlp: MLP
    private let fourier: FourierParams
    private let renderUniformsBuffer: MTLBuffer
    private let tensorArgumentsBuffer: MTLBuffer
    private let layerCountBuffer: MTLBuffer
    private let sigmaBuffer: MTLBuffer
    private let numSamplesBuffer: MTLBuffer

    private let batchSize: Int = 32

    init(device: MTLDevice, library: MTLLibrary, compiler: MTL4Compiler, queue: MTL4CommandQueue) throws {
        let functionDescriptor = MTL4LibraryFunctionDescriptor()
        functionDescriptor.name = "fourierMLP"
        functionDescriptor.library = library
        let pipelineDescriptor = MTL4ComputePipelineDescriptor()
        pipelineDescriptor.computeFunctionDescriptor = functionDescriptor
        self.pipelineState = try compiler.makeComputePipelineState(descriptor: pipelineDescriptor)
        self.pipelineState.reflection?.bindings.forEach { _ in }
        let coopFunctionDescriptor = MTL4LibraryFunctionDescriptor()
        coopFunctionDescriptor.name = "fourierMLPCoop"
        coopFunctionDescriptor.library = library
        let coopDescriptor = MTL4ComputePipelineDescriptor()
        coopDescriptor.computeFunctionDescriptor = coopFunctionDescriptor
        self.cooperativePipelineState = try compiler.makeComputePipelineState(descriptor: coopDescriptor)

        let coopBufferFunctionDescriptor = MTL4LibraryFunctionDescriptor()
        coopBufferFunctionDescriptor.name = "fourierMLPCoopBuffer"
        coopBufferFunctionDescriptor.library = library
        let coopBufferDescriptor = MTL4ComputePipelineDescriptor()
        coopBufferDescriptor.computeFunctionDescriptor = coopBufferFunctionDescriptor
        self.cooperativeBufferPipelineState = try compiler.makeComputePipelineState(descriptor: coopBufferDescriptor)
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
            mlpTensorArguments.mlp.weight[i] = mlp.layers[i].weightTensor.gpuResourceID
        }
        for i in 0..<mlp.layers.count {
            mlpTensorArguments.mlp.bias[i] = mlp.layers[i].biasTensor.gpuResourceID
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

        let trainingWidth = UInt32(fourierModel.metadata?.image?.width ?? 0)
        let trainingHeight = UInt32(fourierModel.metadata?.image?.height ?? 0)

        guard let renderUniformsBuffer = device.makeBuffer(length: MemoryLayout<RenderUniforms>.stride, options: .storageModeShared) else {
            throw FourierEncoderError.failedToCreateBuffer
        }
        var uniforms = RenderUniforms(
            time: 0,
            trainingWidth: trainingWidth,
            trainingHeight: trainingHeight,
            padding: 0
        )
        memcpy(renderUniformsBuffer.contents(), &uniforms, MemoryLayout<RenderUniforms>.stride)
        self.renderUniformsBuffer = renderUniformsBuffer
        
        guard let sigmaBuffer = device.makeBuffer(length: MemoryLayout<Float>.stride, options: .storageModeShared) else {
            throw FourierEncoderError.failedToCreateBuffer
        }
        var sigmaCopy = sigma
        memcpy(sigmaBuffer.contents(), &sigmaCopy, MemoryLayout<Float>.stride)
        self.sigmaBuffer = sigmaBuffer

        guard let numSamplesBuffer = device.makeBuffer(length: MemoryLayout<UInt32>.stride, options: .storageModeShared) else {
            throw FourierEncoderError.failedToCreateBuffer
        }
        self.numSamplesBuffer = numSamplesBuffer

        self.argumentTable.setAddress(tensorArgumentsBuffer.gpuAddress, index: 0)
        self.argumentTable.setAddress(layerCountBuffer.gpuAddress, index: 1)
        self.argumentTable.setAddress(sigmaBuffer.gpuAddress, index: 2)
        self.argumentTable.setAddress(renderUniformsBuffer.gpuAddress, index: 3)

        let bufferTableDesc = MTL4ArgumentTableDescriptor()
        bufferTableDesc.maxBufferBindCount = 6
        self.bufferArgumentTable = try device.makeArgumentTable(descriptor: bufferTableDesc)
        bufferArgumentTable.setAddress(tensorArgumentsBuffer.gpuAddress, index: 0)
        bufferArgumentTable.setAddress(layerCountBuffer.gpuAddress, index: 1)
        bufferArgumentTable.setAddress(sigmaBuffer.gpuAddress, index: 2)
        bufferArgumentTable.setAddress(numSamplesBuffer.gpuAddress, index: 5)

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
        residency.addAllocation(renderUniformsBuffer)
        residency.addAllocation(sigmaBuffer)
        residency.addAllocation(numSamplesBuffer)

        residency.commit()
        self.residencySet = residency
    }

    func supports(_ mode: RenderMode) -> Bool {
        switch mode {
        case .perPixel, .cooperative:
            return true
        }
    }

    func encode(drawableTexture: MTLTexture, commandBuffer: MTL4CommandBuffer, mode: RenderMode) {
        let uniformsPointer = renderUniformsBuffer
            .contents()
            .bindMemory(to: RenderUniforms.self, capacity: 1)
        uniformsPointer.pointee.time = Float(CACurrentMediaTime())

        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        commandBuffer.useResidencySet(residencySet)
        encoder.setArgumentTable(argumentTable)
        argumentTable.setTexture(drawableTexture.gpuResourceID, index: 0)

        switch mode {
        case .perPixel:
            encoder.setComputePipelineState(pipelineState)
            encoder.dispatchThreads(
                threadsPerGrid: .init(width: drawableTexture.width, height: drawableTexture.height, depth: 1),
                threadsPerThreadgroup: .init(width: 16, height: 16, depth: 1)
            )

        case .cooperative:
            encoder.setComputePipelineState(cooperativePipelineState)
            let numPixels = drawableTexture.width * drawableTexture.height
            let threadsPerThreadgroup = MTLSize(
                width: cooperativePipelineState.threadExecutionWidth * 4,
                height: 1,
                depth: 1
            )
            let batches = max(1, (numPixels + batchSize - 1) / batchSize)
            let threadgroupsPerGrid = MTLSize(width: batches, height: 1, depth: 1)
            encoder.dispatchThreadgroups(
                threadgroupsPerGrid: threadgroupsPerGrid,
                threadsPerThreadgroup: threadsPerThreadgroup
            )
        }
        encoder.endEncoding()
    }

    func encodeCooperativeBuffer(
        positions: MTLBuffer,
        outputs: MTLBuffer,
        numSamples: UInt32,
        commandBuffer: MTL4CommandBuffer
    ) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        commandBuffer.useResidencySet(residencySet)
        encoder.setComputePipelineState(cooperativeBufferPipelineState)
        encoder.setArgumentTable(bufferArgumentTable)
        bufferArgumentTable.setAddress(positions.gpuAddress, index: 3)
        bufferArgumentTable.setAddress(outputs.gpuAddress, index: 4)
        var count = numSamples
        memcpy(numSamplesBuffer.contents(), &count, MemoryLayout<UInt32>.stride)

        let threadsPerThreadgroup = MTLSize(
            width: cooperativeBufferPipelineState.threadExecutionWidth * 4,
            height: 1,
            depth: 1
        )
        let batches = max(1, (Int(numSamples) + batchSize - 1) / batchSize)
        let threadgroupsPerGrid = MTLSize(width: batches, height: 1, depth: 1)
        encoder.dispatchThreadgroups(
            threadgroupsPerGrid: threadgroupsPerGrid,
            threadsPerThreadgroup: threadsPerThreadgroup
        )
        encoder.endEncoding()
    }
}
