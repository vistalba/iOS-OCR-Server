import Foundation
import LlamaSwift

actor LLMEngine {
    private var model: OpaquePointer?
    private var context: OpaquePointer?
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

        guard let loaded = llama_load_model_from_file(path, mutableParams) else {
            throw LLMError.failedToInitialize
        }
        self.model = loaded
        self.modelPath = path
        self.modelName = (path as NSString).lastPathComponent

        let ctxParams = llama_context_default_params()
        var mutableCtx = ctxParams
        mutableCtx.n_ctx = UInt32(contextSize)

        guard let ctx = llama_new_context_with_model(loaded, mutableCtx) else {
            llama_free_model(loaded)
            throw LLMError.failedToInitialize
        }
        self.context = ctx
        self.isLoaded = true
    }

    func unloadModel() {
        if let ctx = context { llama_free(ctx) }
        if let m = model { llama_free_model(m) }
        context = nil
        model = nil
        isLoaded = false
        modelPath = nil
        modelName = nil
    }

    func generate(prompt: String, maxTokens: Int32 = 512, temperature: Float = 0.7, topP: Float = 0.9) async throws -> GenerationResult {
        guard let ctx = context, let m = model else {
            throw LLMError.modelNotLoaded
        }

        let nCtx = llama_n_ctx(ctx)
        let tokens = tokenize(text: prompt, addBos: true)
        guard !tokens.isEmpty else { throw LLMError.generationFailed("Failed to tokenize prompt") }

        var batch = llama_batch_init(Int32(tokens.count + maxTokens), 0, 1)
        defer { llama_batch_free(batch) }

        for (i, token) in tokens.enumerated() {
            batch.token[i] = token
            batch.pos[i] = Int32(i)
            batch.seq_id[i] = UnsafeMutablePointer(mutating: [0 as llama_seq_id])
            batch.n_tokens = Int32(i + 1)
        }

        guard llama_decode(ctx, batch) == 0 else {
            throw LLMError.generationFailed("Failed to evaluate prompt")
        }

        var generatedText = ""
        var count: Int32 = 0
        var lastToken = tokens.last ?? 0
        let nLen = Int32(tokens.count)

        while count < maxTokens {
            guard nLen + count < nCtx else { break }

            let newTokenId = sample(ctx: ctx, temperature: temperature, topP: topP)

            if newTokenId == llama_token_eos() { break }

            if let piece = llama_token_to_piece(ctx, newTokenId) {
                generatedText += piece
            }

            llama_kv_cache_seq_rm(ctx, 0, nLen + count, -1)
            var newBatch = llama_batch_init(1, 0, 1)
            newBatch.token[0] = newTokenId
            newBatch.pos[0] = nLen + count
            newBatch.seq_id[0] = UnsafeMutablePointer(mutating: [0 as llama_seq_id])
            newBatch.n_tokens = 1
            defer { llama_batch_free(newBatch) }

            guard llama_decode(ctx, newBatch) == 0 else { break }
            lastToken = newTokenId
            count += 1

            if Task.isCancelled { break }
        }

        return GenerationResult(text: generatedText, tokensGenerated: Int(count), finishReason: count >= maxTokens ? "length" : "stop")
    }

    func generateStream(prompt: String, maxTokens: Int32 = 512, temperature: Float = 0.7, topP: Float = 0.9) -> AsyncThrowingStream<String, Error> {
        AsyncThrowingStream { [self] continuation in
            Task {
                do {
                    let batchSize = 4
                    var buffer = ""

                    let sendBuffer = { (text: String) in
                        if !text.isEmpty {
                            continuation.yield(text)
                        }
                    }

                    let result = try await generate(prompt: prompt, maxTokens: maxTokens, temperature: temperature, topP: topP)
                    sendBuffer(result.text)
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }

    private func tokenize(text: String, addBos: Bool) -> [llama_token] {
        let nTokens = text.count + 3
        var tokens = [llama_token](repeating: 0, count: nTokens)
        let count = llama_tokenize(model, text, Int32(nTokens), &tokens, addBos, false)
        guard count > 0 else { return [] }
        return Array(tokens.prefix(Int(count)))
    }

    private func sample(ctx: OpaquePointer, temperature: Float, topP: Float) -> llama_token {
        let nVocab = llama_n_vocab(model)
        var logits = [Float](repeating: 0, count: Int(nVocab))
        logits.withUnsafeMutableBufferPointer { ptr in
            llama_get_logits_ith(ctx, -1, ptr.baseAddress)
        }

        if temperature > 0 {
            for i in 0..<Int(nVocab) {
                logits[i] /= temperature
            }
        }

        var candidates = [llama_token_data]()
        for i in 0..<Int(nVocab) {
            candidates.append(llama_token_data(id: llama_token(i), logit: logits[i], p: 0))
        }

        var candidatesArray = llama_token_data_array(
            data: &candidates,
            size: Int(nVocab),
            sorted: false
        )

        llama_sample_top_p(nil, &candidatesArray, topP, 1)
        let idx = llama_sample_token(nil, &candidatesArray)
        return idx
    }
}
