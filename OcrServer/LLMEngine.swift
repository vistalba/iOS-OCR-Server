import Foundation
import LlamaSwift

actor LLMEngine {
    private var model: OpaquePointer?
    private var context: OpaquePointer?
    private var vocab: OpaquePointer?
    private var isLoaded = false

    var isModelLoaded: Bool { isLoaded }
    private(set) var modelPath: String?
    private(set) var modelName: String?

    struct GenerationResult {
        let text: String
        let tokensGenerated: Int
        let finishReason: String
    }

    enum LLMError: LocalizedError {
        case modelNotFound(String)
        case modelNotLoaded
        case failedToInitialize
        case generationFailed(String)

        var errorDescription: String? {
            switch self {
            case .modelNotFound(let path): return "Model not found at \(path)"
            case .modelNotLoaded: return "No model is loaded"
            case .failedToInitialize: return "Failed to initialize the model"
            case .generationFailed(let msg): return msg
            }
        }
    }

    func loadModel(path: String, nGpuLayers: Int32 = 99, contextSize: Int32 = 4096) throws {
        guard FileManager.default.fileExists(atPath: path) else {
            throw LLMError.modelNotFound(path)
        }

        let modelParams = llama_model_default_params()
        var mutableParams = modelParams
        mutableParams.n_gpu_layers = nGpuLayers

        guard let loaded = llama_model_load_from_file(path, mutableParams) else {
            throw LLMError.failedToInitialize
        }
        self.model = loaded
        self.vocab = llama_model_get_vocab(loaded)
        self.modelPath = path
        self.modelName = (path as NSString).lastPathComponent

        var ctxParams = llama_context_default_params()
        ctxParams.n_ctx = UInt32(contextSize)

        guard let ctx = llama_init_from_model(loaded, ctxParams) else {
            llama_model_free(loaded)
            throw LLMError.failedToInitialize
        }
        self.context = ctx
        self.isLoaded = true
    }

    func unloadModel() {
        if let ctx = context { llama_free(ctx) }
        if let m = model { llama_model_free(m) }
        context = nil
        model = nil
        vocab = nil
        isLoaded = false
        modelPath = nil
        modelName = nil
    }

    func generate(prompt: String, maxTokens: Int32 = 512, temperature: Float = 0.7, topP: Float = 0.9) async throws -> GenerationResult {
        guard let ctx = context, let vocab = vocab else {
            throw LLMError.modelNotLoaded
        }

        let nCtx = llama_n_ctx(ctx)
        let tokens = tokenize(text: prompt, addBos: true)
        guard !tokens.isEmpty else { throw LLMError.generationFailed("Failed to tokenize prompt") }

        var promptTokens = tokens
        var batch = llama_batch_get_one(&promptTokens, Int32(tokens.count))

        guard llama_decode(ctx, batch) == 0 else {
            throw LLMError.generationFailed("Failed to evaluate prompt")
        }

        var generatedText = ""
        var count: Int32 = 0
        let nLen = Int32(tokens.count)

        while count < maxTokens {
            guard nLen + count < nCtx else { break }

            let newTokenId = sample(ctx: ctx, vocab: vocab, temperature: temperature, topP: topP)

            if newTokenId == llama_vocab_eos(vocab) { break }

            var buffer = [CChar](repeating: 0, count: 16)
            let length = llama_token_to_piece(vocab, newTokenId, &buffer, Int32(buffer.count), 0, false)
            if length > 0 {
                generatedText += String(cString: buffer)
            }

            var nextTokens = [newTokenId]
            var nextBatch = llama_batch_get_one(&nextTokens, 1)

            guard llama_decode(ctx, nextBatch) == 0 else { break }
            count += 1

            if Task.isCancelled { break }
        }

        return GenerationResult(text: generatedText, tokensGenerated: Int(count), finishReason: count >= maxTokens ? "length" : "stop")
    }

    func generateStream(prompt: String, maxTokens: Int32 = 512, temperature: Float = 0.7, topP: Float = 0.9) -> AsyncThrowingStream<String, Error> {
        AsyncThrowingStream { [self] continuation in
            Task {
                do {
                    let result = try await generate(prompt: prompt, maxTokens: maxTokens, temperature: temperature, topP: topP)
                    if !result.text.isEmpty {
                        continuation.yield(result.text)
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }

    private func tokenize(text: String, addBos: Bool) -> [llama_token] {
        guard let vocab = vocab else { return [] }
        let utf8Count = text.utf8.count
        let maxTokens = utf8Count + 3
        var tokens = [llama_token](repeating: 0, count: maxTokens)
        let count = llama_tokenize(vocab, text, Int32(utf8Count), &tokens, Int32(maxTokens), addBos, true)
        guard count > 0 else { return [] }
        return Array(tokens.prefix(Int(count)))
    }

    private func sample(ctx: OpaquePointer, vocab: OpaquePointer, temperature: Float, topP: Float) -> llama_token {
        guard let logits = llama_get_logits_ith(ctx, -1) else {
            return 0
        }
        let nVocab = llama_vocab_n_tokens(vocab)

        if temperature <= 0 || topP <= 0 {
            var maxToken: llama_token = 0
            var maxVal: Float = logits[0]
            for i in 1..<Int(nVocab) {
                if logits[i] > maxVal {
                    maxVal = logits[i]
                    maxToken = llama_token(i)
                }
            }
            return maxToken
        }

        var scaledLogits = [Float](repeating: 0, count: Int(nVocab))
        for i in 0..<Int(nVocab) {
            scaledLogits[i] = logits[i] / temperature
        }

        var maxLogit = scaledLogits[0]
        for i in 1..<Int(nVocab) {
            if scaledLogits[i] > maxLogit { maxLogit = scaledLogits[i] }
        }
        var sum: Float = 0
        for i in 0..<Int(nVocab) {
            scaledLogits[i] = exp(scaledLogits[i] - maxLogit)
            sum += scaledLogits[i]
        }
        for i in 0..<Int(nVocab) {
            scaledLogits[i] /= sum
        }

        var indices = Array(0..<Int(nVocab))
        indices.sort { scaledLogits[$0] > scaledLogits[$1] }

        var cumulative: Float = 0
        var cutoff = Int(nVocab)
        for i in 0..<Int(nVocab) {
            cumulative += scaledLogits[indices[i]]
            if cumulative > topP {
                cutoff = i + 1
                break
            }
        }

        let topIndices = Array(indices.prefix(cutoff))
        let topProbs = topIndices.map { scaledLogits[$0] }
        let totalProb = topProbs.reduce(0, +)
        var r = Float.random(in: 0..<totalProb)
        for i in 0..<topIndices.count {
            r -= topProbs[i]
            if r <= 0 {
                return llama_token(topIndices[i])
            }
        }
        return llama_token(topIndices.last ?? 0)
    }
}
