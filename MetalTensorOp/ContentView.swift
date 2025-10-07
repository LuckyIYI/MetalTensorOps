import SwiftUI
import UniformTypeIdentifiers

enum AppMode: String, CaseIterable, Identifiable {
    case training = "Training"
    case render = "Render"

    var id: String { rawValue }
}

struct ContentView: View {
    @StateObject private var trainingCoordinator = SirenTrainingCoordinator()
    @StateObject private var renderCoordinator = RenderCoordinator()

    @State private var showingImporter = false
    @State private var appMode: AppMode = .training
    @State private var modelKind: ModelKind = .siren
    @State private var selectedRenderMode: RenderMode = .perPixel

    var body: some View {
        VStack(spacing: 16) {
            modePicker

            switch appMode {
            case .training:
                trainingControls
                trainingPane
            case .render:
                renderControls
                renderPane
            }
        }
        .padding(.vertical)
        .fileImporter(isPresented: $showingImporter, allowedContentTypes: [.image]) { result in
            switch result {
            case .success(let url):
                trainingCoordinator.loadImage(url: url)
            case .failure(let error):
                print("[ContentView] File import error: \(error)")
            }
        }
        .onAppear {
            renderCoordinator.setModelKind(modelKind)
            renderCoordinator.setRenderMode(selectedRenderMode)
        }
        .onChange(of: appMode) { mode in
            if mode == .render {
                renderCoordinator.setModelKind(modelKind)
                renderCoordinator.setRenderMode(selectedRenderMode)
            }
        }
        .onChange(of: modelKind) { renderCoordinator.setModelKind($0) }
        .onChange(of: selectedRenderMode) { renderCoordinator.setRenderMode($0) }
    }

    private var modePicker: some View {
        HStack {
            Picker("Mode", selection: $appMode) {
                ForEach(AppMode.allCases) { mode in
                    Text(mode.rawValue).tag(mode)
                }
            }
            .pickerStyle(.segmented)
            Spacer()
        }
        .padding(.horizontal)
    }

    private var trainingControls: some View {
        HStack {
            Button("Load Image") { showingImporter = true }
            Button(trainingCoordinator.isTraining ? "Pause" : "Start") {
                trainingCoordinator.toggleTraining()
            }
            .disabled(trainingCoordinator.groundTruthImage == nil)

            Spacer()

            Text("Loss: \(trainingCoordinator.loss, format: .number.precision(.fractionLength(5)))")
                .monospaced()
        }
        .padding(.horizontal)
    }

    private var trainingPane: some View {
        HStack(spacing: 16) {
            ZStack {
                Color.black
                if let image = trainingCoordinator.groundTruthImage {
                    Image(decorative: image, scale: 1.0)
                        .resizable()
                        .aspectRatio(trainingCoordinator.aspectRatio, contentMode: .fit)
                } else {
                    Text("Ground Truth")
                        .foregroundStyle(.secondary)
                }
            }

            ZStack {
                Color.black
                SirenTrainingMetalView(coordinator: trainingCoordinator)
                    .aspectRatio(trainingCoordinator.aspectRatio, contentMode: .fit)
            }
        }
        .frame(minHeight: 360)
        .padding(.horizontal)
    }

    private var renderControls: some View {
        HStack {
            Picker("Model", selection: $modelKind) {
                ForEach(ModelKind.allCases) { model in
                    Text(model.rawValue).tag(model)
                }
            }
            .pickerStyle(.segmented)

            Picker("Render Mode", selection: $selectedRenderMode) {
                ForEach(RenderMode.allCases) { mode in
                    Text(mode.rawValue).tag(mode)
                }
            }
            .pickerStyle(.segmented)
            .frame(maxWidth: 220)

            Spacer()
        }
        .padding(.horizontal)
    }

    private var renderPane: some View {
        ZStack {
            Color.black
            RenderMetalView(coordinator: renderCoordinator)
                .aspectRatio(renderCoordinator.aspectRatio, contentMode: .fit)
            if !renderCoordinator.isReady {
                ProgressView()
                    .progressViewStyle(.circular)
            }
        }
        .frame(minHeight: 360)
        .padding(.horizontal)
    }
}

#Preview {
    ContentView()
}
