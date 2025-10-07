import Metal
import Foundation
import QuartzCore

enum SirenEncoderError: Error {
    case failedToLocateModelJson(String)
    case noMLPFoundInModelFile
    case failedToCreateBuffer
}

final class SirenEncoder: ComputeEncoder {
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
    private(set) var mlp: MLP
    let renderUniformsBuffer: MTLBuffer
    let tensorArgumentsBuffer: MTLBuffer
    let layerCountBuffer: MTLBuffer
    let numSamplesBuffer: MTLBuffer

    private let batchSize: Int = 32

    init(
        device: MTLDevice,
        library: MTLLibrary,
        compiler: MTL4Compiler,
        queue: MTL4CommandQueue,
        mlp: MLP,
        metadata: Metadata?,
        trainingDimensions: (width: Int, height: Int)? = nil
    ) throws {
        let functionDescriptor = MTL4LibraryFunctionDescriptor()
        functionDescriptor.name = "sirenMLP"
        functionDescriptor.library = library

        let pipelineDescriptor = MTL4ComputePipelineDescriptor()
        pipelineDescriptor.computeFunctionDescriptor = functionDescriptor

        self.pipelineState = try compiler.makeComputePipelineState(descriptor: pipelineDescriptor)

        let coopFunctionDescriptor = MTL4LibraryFunctionDescriptor()
        coopFunctionDescriptor.name = "sirenMLPCoop"
        coopFunctionDescriptor.library = library
        let coopPipelineDescriptor = MTL4ComputePipelineDescriptor()
        coopPipelineDescriptor.computeFunctionDescriptor = coopFunctionDescriptor
        self.cooperativePipelineState = try compiler.makeComputePipelineState(descriptor: coopPipelineDescriptor)

        let coopBufferFunctionDescriptor = MTL4LibraryFunctionDescriptor()
        coopBufferFunctionDescriptor.name = "sirenMLPCoopBuffer"
        coopBufferFunctionDescriptor.library = library
        let coopBufferPipelineDescriptor = MTL4ComputePipelineDescriptor()
        coopBufferPipelineDescriptor.computeFunctionDescriptor = coopBufferFunctionDescriptor
        self.cooperativeBufferPipelineState = try compiler.makeComputePipelineState(descriptor: coopBufferPipelineDescriptor)

        let tableDesc = MTL4ArgumentTableDescriptor()
        tableDesc.maxTextureBindCount = 1
        tableDesc.maxBufferBindCount = 4

        self.argumentTable = try device.makeArgumentTable(descriptor: tableDesc)

        var mlpTensorArguments = MetalMLPTensorArguments()
        for i in 0..<mlp.layers.count {
            mlpTensorArguments.weight[i] = mlp.layers[i].weightTensor.gpuResourceID
        }
        for i in 0..<mlp.layers.count {
            mlpTensorArguments.bias[i] = mlp.layers[i].biasTensor.gpuResourceID
        }

        guard let tensorArgumentsBuffer = device.makeBuffer(length: MemoryLayout<MetalMLPTensorArguments>.stride, options: .storageModeShared) else {
            throw SirenEncoderError.failedToCreateBuffer
        }
        self.tensorArgumentsBuffer = tensorArgumentsBuffer
        memcpy(tensorArgumentsBuffer.contents(), &mlpTensorArguments, MemoryLayout<MetalMLPTensorArguments>.stride)

        var layerCount32 = UInt32(mlp.layers.count)
        guard let layerCountBuffer = device.makeBuffer(length: MemoryLayout<UInt32>.size, options: .storageModeShared) else {
            throw SirenEncoderError.failedToCreateBuffer
        }
        layerCountBuffer.contents().copyMemory(from: &layerCount32, byteCount: MemoryLayout<UInt32>.size)
        self.layerCountBuffer = layerCountBuffer

        let trainingWidth = UInt32(trainingDimensions?.width ?? metadata?.image?.width ?? 0)
        let trainingHeight = UInt32(trainingDimensions?.height ?? metadata?.image?.height ?? 0)

        guard let renderUniformsBuffer = device.makeBuffer(length: MemoryLayout<RenderUniforms>.stride, options: .storageModeShared) else {
            throw SirenEncoderError.failedToCreateBuffer
        }
        var uniforms = RenderUniforms(
            time: 0,
            trainingWidth: trainingWidth,
            trainingHeight: trainingHeight,
            padding: 0
        )
        memcpy(renderUniformsBuffer.contents(), &uniforms, MemoryLayout<RenderUniforms>.stride)
        self.renderUniformsBuffer = renderUniformsBuffer

        guard let numSamplesBuffer = device.makeBuffer(length: MemoryLayout<UInt32>.stride, options: .storageModeShared) else {
            throw SirenEncoderError.failedToCreateBuffer
        }
        var zero: UInt32 = 0
        numSamplesBuffer.contents().copyMemory(from: &zero, byteCount: MemoryLayout<UInt32>.stride)
        self.numSamplesBuffer = numSamplesBuffer

        self.argumentTable.setAddress(tensorArgumentsBuffer.gpuAddress, index: 0)
        self.argumentTable.setAddress(layerCountBuffer.gpuAddress, index: 1)
        self.argumentTable.setAddress(renderUniformsBuffer.gpuAddress, index: 2)

        let bufferTableDesc = MTL4ArgumentTableDescriptor()
        bufferTableDesc.maxBufferBindCount = 5
        self.bufferArgumentTable = try device.makeArgumentTable(descriptor: bufferTableDesc)
        bufferArgumentTable.setAddress(tensorArgumentsBuffer.gpuAddress, index: 0)
        bufferArgumentTable.setAddress(layerCountBuffer.gpuAddress, index: 1)
        bufferArgumentTable.setAddress(numSamplesBuffer.gpuAddress, index: 4)

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
        residency.addAllocation(tensorArgumentsBuffer)
        residency.addAllocation(renderUniformsBuffer)
        residency.addAllocation(numSamplesBuffer)

        residency.commit()
        self.residencySet = residency
        self.mlp = mlp
    }

    convenience init(device: MTLDevice, library: MTLLibrary, compiler: MTL4Compiler, queue: MTL4CommandQueue) throws {
        let fileName = "siren"

        guard let url = Bundle.main.url(forResource: fileName, withExtension: "json") else {
            throw SirenEncoderError.failedToLocateModelJson(fileName)
        }
        let data = try Data(contentsOf: url)

        let sirenModel = try JSONDecoder().decode(SirenModel.self, from: data)
        guard let mlp = sirenModel.mlp else {
            throw SirenEncoderError.noMLPFoundInModelFile
        }

        try self.init(device: device, library: library, compiler: compiler, queue: queue, mlp: mlp, metadata: sirenModel.metadata)
    }

    func supports(_ mode: RenderMode) -> Bool {
        switch mode {
        case .perPixel, .cooperative:
            return true
        }
    }

    func updateTrainingDimensions(width: Int, height: Int) {
        let pointer = renderUniformsBuffer
            .contents()
            .bindMemory(to: RenderUniforms.self, capacity: 1)
        pointer.pointee.trainingWidth = UInt32(max(width, 0))
        pointer.pointee.trainingHeight = UInt32(max(height, 0))
    }

    func setNumSamples(_ count: Int) {
        var value = UInt32(max(count, 0))
        numSamplesBuffer.contents().copyMemory(from: &value, byteCount: MemoryLayout<UInt32>.stride)
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
        bufferArgumentTable.setAddress(positions.gpuAddress, index: 2)
        bufferArgumentTable.setAddress(outputs.gpuAddress, index: 3)
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
