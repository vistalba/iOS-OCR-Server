import Foundation
import Combine

@MainActor
final class LLMManager: ObservableObject {
    static let shared = LLMManager()

    private let engine = LLMEngine()

    @Published var status: String = "idle"
    @Published var isModelLoaded: Bool = false
    @Published var availableModels: [URL] = []
    @Published var selectedModelPath: URL?
    @Published var isLoadingModel: Bool = false

    // Settings from UserDefaults
    var autoLoadModel: Bool {
        get { UserDefaults.standard.bool(forKey: "llm_autoLoad") }
        set { UserDefaults.standard.set(newValue, forKey: "llm_autoLoad") }
    }

    var contextSize: Int32 {
        get { Int32(UserDefaults.standard.integer(forKey: "llm_contextSize").coerced(to: 2048...8192, default: 4096)) }
        set { UserDefaults.standard.set(Int(newValue), forKey: "llm_contextSize") }
    }

    var gpuLayers: Int32 {
        get { Int32(UserDefaults.standard.integer(forKey: "llm_gpuLayers").coerced(to: 0...99, default: 99)) }
        set { UserDefaults.standard.set(Int(newValue), forKey: "llm_gpuLayers") }
    }

    var temperature: Float {
        get { UserDefaults.standard.float(forKey: "llm_temperature").coerced(to: 0.0...2.0, default: 0.7) }
        set { UserDefaults.standard.set(newValue, forKey: "llm_temperature") }
    }

    var maxTokens: Int32 {
        get { Int32(UserDefaults.standard.integer(forKey: "llm_maxTokens").coerced(to: 64...4096, default: 512)) }
        set { UserDefaults.standard.set(Int(newValue), forKey: "llm_maxTokens") }
    }

    private init() {
        refreshAvailableModels()
        if autoLoadModel, let lastPath = UserDefaults.standard.url(forKey: "llm_lastModelPath") {
            selectedModelPath = lastPath
            if FileManager.default.fileExists(atPath: lastPath.path) {
                Task { try? await loadModel(path: lastPath) }
            }
        }
    }

    func refreshAvailableModels() {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        let contents = (try? FileManager.default.contentsOfDirectory(at: docs, includingPropertiesForKeys: nil)) ?? []
        availableModels = contents.filter { $0.pathExtension == "gguf" }.sorted { $0.lastPathComponent < $1.lastPathComponent }
    }

    func loadModel(path: URL) async throws {
        isLoadingModel = true
        status = "loading model..."
        try await engine.loadModel(path: path.path, nGpuLayers: gpuLayers, contextSize: contextSize)
        isModelLoaded = true
        selectedModelPath = path
        UserDefaults.standard.set(path, forKey: "llm_lastModelPath")
        status = "model loaded: \(path.lastPathComponent)"
        isLoadingModel = false
    }

    func loadModel() async throws {
        guard let path = selectedModelPath else { return }
        try await loadModel(path: path)
    }

    func unloadModel() {
        engine.unloadModel()
        isModelLoaded = false
        status = "model unloaded"
        selectedModelPath = nil
    }

    func generate(prompt: String) async throws -> String {
        status = "generating..."
        let result = try await engine.generate(
            prompt: prompt,
            maxTokens: maxTokens,
            temperature: temperature,
            topP: 0.9
        )
        status = "ready"
        return result.text
    }

    func generateStream(prompt: String) -> AsyncThrowingStream<String, Error> {
        status = "streaming..."
        return engine.generateStream(
            prompt: prompt,
            maxTokens: maxTokens,
            temperature: temperature,
            topP: 0.9
        )
    }

    func modelSizeBytes(for url: URL) -> Int64 {
        (try? FileManager.default.attributesOfItem(atPath: url.path)[.size] as? Int64) ?? 0
    }

    func deleteModel(_ url: URL) {
        try? FileManager.default.removeItem(at: url)
        refreshAvailableModels()
        if selectedModelPath == url { selectedModelPath = nil }
    }
}

extension Int {
    func coerced(to range: ClosedRange<Int>, default: Int) -> Int {
        self < range.lowerBound ? range.lowerBound : (self > range.upperBound ? range.upperBound : self)
    }
}

extension Float {
    func coerced(to range: ClosedRange<Float>, default: Float) -> Float {
        self < range.lowerBound ? range.lowerBound : (self > range.upperBound ? range.upperBound : self)
    }
}
