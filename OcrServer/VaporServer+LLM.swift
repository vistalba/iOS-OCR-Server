import Vapor

extension VaporServer {
    func routesLLM(_ app: Application) throws {
        app.get("v1", "models") { req async throws -> Response in
            let manager = await LLMManager.shared
            let models: [[String: Any]] = await manager.availableModels.map { url in
                ["id": url.lastPathComponent, "object": "model", "created": Int(Date().timeIntervalSince1970), "owned_by": "local"]
            }
            return try Self.jsonResponse(.ok, ["object": "list", "data": models])
        }

        app.on(.POST, "v1", "chat", "completions", body: .collect(maxSize: "10mb")) { req async throws -> Response in
            let manager = await LLMManager.shared
            struct ChatRequest: Content { var model: String?; var messages: [ChatMessage]?; var temperature: Double?; var maxTokens: Int?; var stream: Bool? }
            struct ChatMessage: Content { var role: String; var content: String }
            guard let chatRequest = try? req.content.decode(ChatRequest.self) else {
                return try Self.jsonResponse(.badRequest, ["error": "Invalid request body"])
            }
            guard await manager.isModelLoaded else {
                return try Self.jsonResponse(.badRequest, ["error": "No model loaded. Load a model first."])
            }
            let prompt = chatRequest.messages?.map { "\($0.role): \($0.content)" }.joined(separator: "\n") ?? ""
            let defaultMaxTokens = await manager.maxTokens
            let defaultTemp = await manager.temperature
            let maxT = Int32(chatRequest.maxTokens ?? Int(defaultMaxTokens))
            let temp = Float(chatRequest.temperature ?? Double(defaultTemp))
            if chatRequest.stream == true {
                return try await self.handleStreaming(prompt: prompt, isChat: true)
            } else {
                return try await self.handleNonStreaming(prompt: prompt, maxTokens: maxT, temperature: temp, isChat: true)
            }
        }

        app.on(.POST, "v1", "completions", body: .collect(maxSize: "10mb")) { req async throws -> Response in
            let manager = await LLMManager.shared
            struct CompletionRequest: Content { var model: String?; var prompt: String?; var temperature: Double?; var maxTokens: Int?; var stream: Bool? }
            guard let compRequest = try? req.content.decode(CompletionRequest.self) else {
                return try Self.jsonResponse(.badRequest, ["error": "Invalid request body"])
            }
            guard await manager.isModelLoaded else {
                return try Self.jsonResponse(.badRequest, ["error": "No model loaded"])
            }
            let prompt = compRequest.prompt ?? ""
            let defaultMaxTokens = await manager.maxTokens
            let defaultTemp = await manager.temperature
            let maxT = Int32(compRequest.maxTokens ?? Int(defaultMaxTokens))
            let temp = Float(compRequest.temperature ?? Double(defaultTemp))
            if compRequest.stream == true {
                return try await self.handleStreaming(prompt: prompt, isChat: false)
            } else {
                return try await self.handleNonStreaming(prompt: prompt, maxTokens: maxT, temperature: temp, isChat: false)
            }
        }
    }

    private func handleNonStreaming(prompt: String, maxTokens: Int32, temperature: Float, isChat: Bool) async throws -> Response {
        let result = try await LLMManager.shared.generate(prompt: prompt)
        let modelName = await LLMManager.shared.selectedModelPath?.lastPathComponent ?? "unknown"
        let base: [String: Any] = [
            "id": isChat ? "chatcmpl-\(UUID().uuidString)" : "cmpl-\(UUID().uuidString)",
            "object": isChat ? "chat.completion" : "text_completion",
            "created": Int(Date().timeIntervalSince1970), "model": modelName,
            "usage": ["prompt_tokens": prompt.count / 4, "completion_tokens": result.count / 4, "total_tokens": (prompt.count + result.count) / 4]
        ]
        var response = base
        if isChat {
            response["choices"] = [["index": 0, "message": ["role": "assistant", "content": result], "finish_reason": "stop"]]
        } else {
            response["choices"] = [["index": 0, "text": result, "finish_reason": "stop"]]
        }
        return try Self.jsonResponse(.ok, response)
    }

    private func handleStreaming(prompt: String, isChat: Bool) async throws -> Response {
        return Response(body: .init(stream: { writer in
            Task {
                let manager = await LLMManager.shared
                let modelName = await manager.selectedModelPath?.lastPathComponent ?? "unknown"
                let stream = manager.generateStream(prompt: prompt)
                do {
                    for try await text in stream {
                        let chunk: [String: Any] = [
                            "id": isChat ? "chatcmpl-\(UUID().uuidString)" : "cmpl-\(UUID().uuidString)",
                            "object": isChat ? "chat.completion.chunk" : "text_completion",
                            "created": Int(Date().timeIntervalSince1970), "model": modelName,
                            "choices": [["index": 0, isChat ? "delta" : "text": text, "finish_reason": "stop"]]
                        ]
                        if let data = try? JSONSerialization.data(withJSONObject: chunk),
                           let str = String(data: data, encoding: .utf8) {
                            try? await writer.write(.buffer(ByteBuffer(string: "data: \(str)\n\n")))
                        }
                    }
                    try? await writer.write(.buffer(ByteBuffer(string: "data: [DONE]\n\n")))
                } catch {}
                try? await writer.write(.end)
            }
        }))
    }
}
