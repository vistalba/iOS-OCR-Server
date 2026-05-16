import SwiftUI

struct LLMSettingsView: View {
    @StateObject private var manager = LLMManager.shared
    @State private var showingModelDownloader = false
    @State private var contextSize: Double = 4096
    @State private var gpuLayers: Double = 99
    @State private var temperature: Double = 0.7
    @State private var maxTokens: Double = 512
    @State private var autoLoad: Bool = false

    var body: some View {
        NavigationView {
            Form {
                Section("Model") {
                    if manager.isModelLoaded {
                        HStack {
                            Image(systemName: "brain.head.profile")
                                .foregroundColor(.green)
                            Text("Model loaded")
                            Spacer()
                            Button("Unload") {
                                Task { await manager.unloadModel() }
                            }
                            .foregroundColor(.red)
                        }

                        if let path = manager.selectedModelPath {
                            Text(path.lastPathComponent)
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                    } else {
                        HStack {
                            Image(systemName: "brain.head.profile")
                                .foregroundColor(.gray)
                            Text("No model loaded")
                        }
                    }

                    if !manager.availableModels.isEmpty {
                        Picker("Select Model", selection: $manager.selectedModelPath) {
                            Text("None").tag(nil as URL?)
                            ForEach(manager.availableModels, id: \.self) { url in
                                HStack {
                                    Text(url.lastPathComponent)
                                    let size = manager.modelSizeBytes(for: url)
                                    Text(ByteCountFormatter.string(fromByteCount: size, countStyle: .file))
                                        .font(.caption)
                                        .foregroundColor(.secondary)
                                }
                                .tag(url as URL?)
                            }
                        }

                        Button(action: loadSelectedModel) {
                            HStack {
                                if manager.isLoadingModel {
                                    ProgressView().scaleEffect(0.8)
                                }
                                Text("Load Model")
                            }
                        }
                        .disabled(manager.isLoadingModel || manager.selectedModelPath == nil)
                    } else {
                        Text("No models found. Download one from HuggingFace.")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }

                    Button("Download Model from HuggingFace") {
                        showingModelDownloader = true
                    }
                }

                Section("Model Lifecycle") {
                    Toggle("Auto-load on app start", isOn: $autoLoad)
                        .onChange(of: autoLoad) { _, v in manager.autoLoadModel = v }
                }

                Section("Inference Parameters") {
                    VStack {
                        HStack {
                            Text("Context Size: \(Int(contextSize))")
                            Spacer()
                        }
                        Slider(value: $contextSize, in: 2048...8192, step: 512)
                            .onChange(of: contextSize) { _, v in manager.contextSize = Int32(v) }
                    }

                    VStack {
                        HStack {
                            Text("GPU Layers: \(Int(gpuLayers))")
                            Spacer()
                        }
                        Slider(value: $gpuLayers, in: 0...99, step: 1)
                            .onChange(of: gpuLayers) { _, v in manager.gpuLayers = Int32(v) }
                    }

                    VStack {
                        HStack {
                            Text("Temperature: \(temperature, specifier: "%.2f")")
                            Spacer()
                        }
                        Slider(value: $temperature, in: 0.0...2.0, step: 0.05)
                            .onChange(of: temperature) { _, v in manager.temperature = Float(v) }
                    }

                    VStack {
                        HStack {
                            Text("Max Tokens: \(Int(maxTokens))")
                            Spacer()
                        }
                        Slider(value: $maxTokens, in: 64...4096, step: 64)
                            .onChange(of: maxTokens) { _, v in manager.maxTokens = Int32(v) }
                    }
                }

                Section("About") {
                    Text("Models should be in GGUF format. Place them in the app's Documents folder via iTunes File Sharing or download them directly.")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
            }
            .navigationTitle("LLM Settings")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Done") { dismiss() }
                }
            }
            .onAppear {
                contextSize = Double(manager.contextSize)
                gpuLayers = Double(manager.gpuLayers)
                temperature = Double(manager.temperature)
                maxTokens = Double(manager.maxTokens)
                autoLoad = manager.autoLoadModel
                manager.refreshAvailableModels()
            }
            .sheet(isPresented: $showingModelDownloader) {
                ModelDownloadView()
            }
        }
    }

    @Environment(\.dismiss) private var dismiss

    private func loadSelectedModel() {
        guard let path = manager.selectedModelPath else { return }
        Task { try? await manager.loadModel(path: path) }
    }
}
