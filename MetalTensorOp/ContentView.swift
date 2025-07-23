//
//  ContentView.swift
//  InstantNGP
//
//  Created by Лаки Ийнбор on 6/15/25.
//

import SwiftUI
import Metal
import MetalKit

#if os(macOS)
import AppKit
#elseif os(iOS)
import UIKit
#endif

#if os(macOS)
struct MetalCircleView: NSViewRepresentable {
    func makeCoordinator() -> Coordinator { Coordinator() }
    func makeNSView(context: Context) -> MTKView {
        let mtkView = MTKView()
        guard let device = MTLCreateSystemDefaultDevice() else { return mtkView }
        mtkView.device = device
        mtkView.framebufferOnly = false
        mtkView.delegate = context.coordinator
        mtkView.clearColor = .init(red: 1, green: 1, blue: 1, alpha: 1)
        context.coordinator.setup(device: device, mtkView: mtkView)
        
        return mtkView
    }
    func updateNSView(_ nsView: MTKView, context: Context) {}
}
#else
struct MetalCircleView: UIViewRepresentable {
    func makeCoordinator() -> Coordinator { Coordinator() }
    func makeUIView(context: Context) -> MTKView {
        let mtkView = MTKView()
        guard let device = MTLCreateSystemDefaultDevice() else { return mtkView }
        mtkView.device = device
        mtkView.framebufferOnly = false
        mtkView.delegate = context.coordinator
        mtkView.clearColor = .init(red: 1, green: 1, blue: 1, alpha: 1)
        context.coordinator.setup(device: device, mtkView: mtkView)
        
        return mtkView
    }
    func updateUIView(_ uiView: MTKView, context: Context) {}
}
#endif

@available(macOS 26.0, iOS 16.0, *)
class Coordinator: NSObject, MTKViewDelegate {
    var device: MTLDevice!
    var commandQueue: MTL4CommandQueue!
    var commandAllocator: MTL4CommandAllocator!
    var encoder: MetalEncoder!
    var commandBuffer: MTL4CommandBuffer!
    var compiler: MTL4Compiler!
    
    func setup(device: MTLDevice, mtkView: MTKView) {
        self.device = device
        commandQueue = device.makeMTL4CommandQueue()
        commandAllocator = device.makeCommandAllocator()
        compiler = try! device.makeCompiler(descriptor: .init())
        let library = try! device.makeDefaultLibrary(bundle: .main)
                
        encoder = try! MetalEncoder(device: device, library: library, compiler: compiler,queue: commandQueue)
        commandBuffer = device.makeCommandBuffer()
        
        commandQueue.addResidencySet((mtkView.layer as? CAMetalLayer)!.residencySet)

        mtkView.drawableSize = CGSize(width: 512, height: 512)
    }
    
    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {}
    
    var done = false
    func draw(in view: MTKView) {
        guard let drawable = view.currentDrawable else { return }
        
        commandAllocator.reset()
        commandBuffer.beginCommandBuffer(allocator: commandAllocator)

        encoder.encode(drawableTexture: drawable.texture, commandBuffer: commandBuffer)
        commandBuffer.endCommandBuffer()
        commandQueue.waitForDrawable(drawable)
        commandQueue.commit([commandBuffer])
        commandQueue.signalDrawable(drawable)
        drawable.present()
    }
}

@available(macOS 26.0, iOS 16.0, *)
struct ContentView: View {
    var body: some View {
        MetalCircleView()
            .frame(width: 512, height: 512)
            .padding()
    }
}

#Preview {
    ContentView()
}
