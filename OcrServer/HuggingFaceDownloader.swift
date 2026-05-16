import Foundation

actor HuggingFaceDownloader {
    struct ModelInfo: Identifiable, Codable {
        let id: String
        let name: String
        let repoId: String
        let filename: String
        let size: Int64
        let quantization: String

        var displaySize: String {
            let formatter = ByteCountFormatter()
            formatter.countStyle = .file
            return formatter.string(fromByteCount: size)
        }
    }

    struct DownloadProgress {
        let receivedBytes: Int64
        let totalBytes: Int64
        let fraction: Double
        let speed: String
    }

    enum DownloadError: LocalizedError {
        case noModelsFound
        case networkError(String)
        case cancelled

        var errorDescription: String? {
            switch self {
            case .noModelsFound: return "No GGUF models found for this repository"
            case .networkError(let msg): return msg
            case .cancelled: return "Download cancelled"
            }
        }
    }

    private var activeTask: URLSessionDownloadTask?
    private var progressContinuation: AsyncStream<DownloadProgress>.Continuation?

    func searchModels(query: String) async throws -> [ModelInfo] {
        let searchQuery = query.addingPercentEncoding(withAllowedCharacters: .urlQueryAllowed) ?? query
        let url = URL(string: "https://huggingface.co/api/models?search=\(searchQuery)&filter=gguf&sort=downloads&direction=-1&limit=20")!

        let (data, _) = try await URLSession.shared.data(from: url)
        let models = try JSONDecoder().decode([HFModelSearchResult].self, from: data)

        var results: [ModelInfo] = []
        for model in models {
            let files = try await listGGUFFiles(repoId: model.id)
            results.append(contentsOf: files)
        }
        return results
    }

    func listGGUFFiles(repoId: String) async throws -> [ModelInfo] {
        let url = URL(string: "https://huggingface.co/api/models/\(repoId)")!
        let (data, _) = try await URLSession.shared.data(from: url)
        let repo = try JSONDecoder().decode(HFRepoDetail.self, from: data)

        return repo.siblings?
            .filter { $0.rfilename.hasSuffix(".gguf") }
            .map { sibling in
                let size = repo.siblingsSize?[sibling.rfilename] ?? 0
                let qType = extractQuantization(from: sibling.rfilename)
                return ModelInfo(
                    id: "\(repoId)/\(sibling.rfilename)",
                    name: sibling.rfilename,
                    repoId: repoId,
                    filename: sibling.rfilename,
                    size: size,
                    quantization: qType
                )
            } ?? []
    }

    func downloadModel(_ model: ModelInfo, to directory: URL) -> AsyncStream<DownloadProgress> {
        AsyncStream { continuation in
            self.progressContinuation = continuation
            Task {
                let fileURL = directory.appendingPathComponent(model.filename)
                let url = URL(string: "https://huggingface.co/\(model.repoId)/resolve/main/\(model.filename)")!

                let session = URLSession(configuration: .default)
                let task = session.dataTask(with: url) { [weak self] data, _, error in
                    guard let data = data else {
                        continuation.finish()
                        return
                    }
                    try? data.write(to: fileURL)
                    continuation.yield(DownloadProgress(
                        receivedBytes: Int64(data.count),
                        totalBytes: Int64(data.count),
                        fraction: 1.0,
                        speed: "done"
                    ))
                    continuation.finish()
                }
                task.resume()
            }
        }
    }

    func cancelDownload() {
        activeTask?.cancel()
        progressContinuation?.finish()
    }

    private func extractQuantization(from filename: String) -> String {
        let patterns = ["Q2_K", "Q3_K", "Q4_K_M", "Q4_K_S", "Q5_K_M", "Q5_K_S", "Q6_K", "Q8_0", "F16", "IQ1_S", "IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ2_M", "IQ3_XXS", "IQ3_XS", "IQ3_S", "IQ3_M", "IQ4_NL", "IQ4_XS"]
        for p in patterns {
            if filename.contains(p) { return p }
        }
        return "unknown"
    }
}

struct HFModelSearchResult: Codable {
    let id: String
    let downloads: Int?
}

struct HFRepoDetail: Codable {
    let id: String
    let siblings: [HFSibling]?
    let siblingsSize: [String: Int64]?

    enum CodingKeys: String, CodingKey {
        case id, siblings
        case siblingsSize = "siblingSize"
    }
}

struct HFSibling: Codable {
    let rfilename: String
}
