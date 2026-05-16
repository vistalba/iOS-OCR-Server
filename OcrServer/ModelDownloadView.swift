import SwiftUI

struct ModelDownloadView: View {
    @Environment(\.dismiss) private var dismiss
    @State private var searchQuery = "Qwen"
    @State private var searchResults: [HuggingFaceDownloader.ModelInfo] = []
    @State private var isSearching = false
    @State private var isDownloading = false
    @State private var downloadProgress: Double = 0
    @State private var statusMessage = ""

    private let downloader = HuggingFaceDownloader()

    var body: some View {
        NavigationView {
            VStack {
                HStack {
                    TextField("Search models...", text: $searchQuery)
                        .textFieldStyle(.roundedBorder)
                        .autocapitalization(.none)
                    Button("Search") { search() }
                        .disabled(isSearching || searchQuery.isEmpty)
                }
                .padding()

                if isSearching {
                    ProgressView("Searching...")
                }

                if isDownloading {
                    VStack(spacing: 8) {
                        ProgressView(value: downloadProgress)
                        Text(statusMessage).font(.caption).foregroundColor(.secondary)
                    }
                    .padding()
                }

                List {
                    ForEach(searchResults) { model in
                        VStack(alignment: .leading, spacing: 4) {
                            Text(model.name).font(.headline)
                            HStack {
                                Text(model.quantization).font(.caption).padding(.horizontal, 6).padding(.vertical, 2)
                                    .background(Color.blue.opacity(0.2)).cornerRadius(4)
                                Text(model.displaySize).font(.caption).foregroundColor(.secondary)
                            }
                        }
                        .swipeActions {
                            Button("Download") {
                                downloadModel(model)
                            }
                            .tint(.blue)
                        }
                    }
                }
            }
            .navigationTitle("Download Model")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Done") { dismiss() }
                }
            }
        }
    }

    private func search() {
        isSearching = true
        Task {
            do {
                searchResults = try await downloader.searchModels(query: searchQuery)
            } catch {
                statusMessage = "Search failed: \(error.localizedDescription)"
            }
            isSearching = false
        }
    }

    private func downloadModel(_ model: HuggingFaceDownloader.ModelInfo) {
        isDownloading = true
        statusMessage = "Downloading \(model.name)..."
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!

        Task {
            for await progress in downloader.downloadModel(model, to: docs) {
                downloadProgress = progress.fraction
                statusMessage = "\(Int(progress.fraction * 100))% - \(progress.speed)"
            }
            isDownloading = false
            statusMessage = "Download complete!"
            try? await Task.sleep(nanoseconds: 1_500_000_000)
            dismiss()
        }
    }
}
