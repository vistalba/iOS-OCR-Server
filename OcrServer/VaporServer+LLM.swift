import Vapor

extension VaporServer {
    func routesLLM(_ app: Application) throws {
        let llmManager = LLMManager.shared

        app.get("v1", "models") { [self] req async throws -> Response in
            let models: [[String: Any]] = await llmManager.availableModels.map { url in
                [
                    "id": url.lastPathComponent,
                    "object": "model",
                    "created": Int(Date().timeIntervalSince1970),
                    "owned_by": "local"
                ]
            }
            return try Self.jsonResponse(.ok, ["object": "list", "data": models])
        }

        app.on(.POST, "v1", "chat", "completions", body: .collect(maxSize: "10mb")) { [self] req async throws -> Response in
            struct ChatRequest: Content {
                var model: String?
                var messages: [ChatMessage]?
                var temperature: Double?
                var maxTokens: Int?
                var stream: Bool?
            }
            struct ChatMessage: Content {
                var role: String
                var content: String
            }

            guard let chatRequest = try? req.content.decode(ChatRequest.self) else {
                return try Self.jsonResponse(.badRequest, ["error": "Invalid request body"])
            }
            guard await llmManager.isModelLoaded else {
                return try Self.jsonResponse(.badRequest, ["error": "No model loaded. Load a model first."])
            }

            let prompt = chatRequest.messages?.map { "\($0.role): \($0.content)" }.joined(separator: "\n") ?? ""
            let maxT = Int32(chatRequest.maxTokens ?? Int(await llmManager.maxTokens))
            let temp = Float(chatRequest.temperature ?? Double(await llmManager.temperature))

            if chatRequest.stream == true {
                return try await self.handleStreaming(prompt: prompt, maxTokens: maxT, temperature: temp, isChat: true)
            } else {
                return try await self.handleNonStreaming(prompt: prompt, maxTokens: maxT, temperature: temp, isChat: true)
            }
        }

        app.on(.POST, "v1", "completions", body: .collect(maxSize: "10mb")) { [self] req async throws -> Response in
            struct CompletionRequest: Content {
                var model: String?
                var prompt: String?
                var temperature: Double?
                var maxTokens: Int?
                var stream: Bool?
            }

            guard let compRequest = try? req.content.decode(CompletionRequest.self) else {
                return try Self.jsonResponse(.badRequest, ["error": "Invalid request body"])
            }
            guard await llmManager.isModelLoaded else {
                return try Self.jsonResponse(.badRequest, ["error": "No model loaded"])
            }

            let prompt = compRequest.prompt ?? ""
            let maxT = Int32(compRequest.maxTokens ?? Int(await llmManager.maxTokens))
            let temp = Float(compRequest.temperature ?? Double(await llmManager.temperature))

            if compRequest.stream == true {
                return try await self.handleStreaming(prompt: prompt, maxTokens: maxT, temperature: temp, isChat: false)
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
            "created": Int(Date().timeIntervalSince1970),
            "model": modelName,
            "usage": [
                "prompt_tokens": prompt.count / 4,
                "completion_tokens": result.count / 4,
                "total_tokens": (prompt.count + result.count) / 4
            ]
        ]
        let choices: [[String: Any]]
        if isChat {
            choices = [
                ["index": 0, "message": ["role": "assistant", "content": result], "finish_reason": "stop"]
            ]
        } else {
            choices = [
                ["index": 0, "text": result, "finish_reason": "stop"]
            ]
        }
        var response = base
        response["choices"] = choices
        return try Self.jsonResponse(.ok, response)
    }

    private func handleStreaming(prompt: String, maxTokens: Int32, temperature: Float, isChat: Bool) async throws -> Response {
        let stream = LLMManager.shared.generateStream(prompt: prompt)
        let modelName = await LLMManager.shared.selectedModelPath?.lastPathComponent ?? "unknown"
        return Response(body: .init(stream: { writer in
            do {
                for try await text in stream {
                    let chunk: [String: Any] = [
                        "id": isChat ? "chatcmpl-\(UUID().uuidString)" : "cmpl-\(UUID().uuidString)",
                        "object": isChat ? "chat.completion.chunk" : "text_completion",
                        "created": Int(Date().timeIntervalSince1970),
                        "model": modelName,
                        "choices": [
                            [
                                "index": 0,
                                isChat ? "delta" : "text": text,
                                "finish_reason": "stop"
                            ]
                        ]
                    ]
                    let jsonData = try JSONSerialization.data(withJSONObject: chunk)
                    if let jsonString = String(data: jsonData, encoding: .utf8) {
                        try await writer.write(.buffer(ByteBuffer(string: "data: \(jsonString)\n\n")))
                    }
                }
                try await writer.write(.buffer(ByteBuffer(string: "data: [DONE]\n\n")))
            } catch {}
            try await writer.write(.end)
        }))
    }
}
