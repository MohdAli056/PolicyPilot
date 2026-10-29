"use client";

import React, { useEffect, useState } from "react";

/** Backend API base URL, configurable via environment variable. */
const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

type HealthStatus = "loading" | "healthy" | "unavailable";

interface HealthData {
  status: string;
  service: string;
  version: string;
  timestamp: string;
}

interface PageExtraction {
  page_number: number;
  text: string;
  character_count: number;
}

interface DocumentMetadata {
  document_id: string;
  filename: string;
  content_type: string;
  file_size_bytes: number;
  total_pages: number;
  total_characters: number;
  sha256_hash: string;
  ingestion_status: string;
  uploaded_at: string;
}

interface DocumentDetail extends DocumentMetadata {
  pages: PageExtraction[];
}

interface DocumentChunk {
  chunk_id: string;
  document_id: string;
  chunk_index: number;
  text: string;
  section: string | null;
  page_numbers: number[];
  character_count: number;
}

type SearchMode = "hybrid" | "semantic";

interface SearchResultItem {
  document_id: string;
  filename: string;
  chunk_id: string;
  chunk_index: number;
  text: string;
  section: string | null;
  page_numbers: number[];
  character_count: number;
  score?: number;
  vector_rank?: number | null;
  keyword_rank?: number | null;
  rrf_score?: number;
  retrievers?: string[];
}

interface IndexState {
  status: "idle" | "indexing" | "indexed" | "empty" | "failed";
  message?: string;
}

export default function Home() {
  // Backend health state
  const [healthStatus, setHealthStatus] = useState<HealthStatus>("loading");
  const [healthData, setHealthData] = useState<HealthData | null>(null);
  const [healthError, setHealthError] = useState<string | null>(null);

  // Documents state
  const [documents, setDocuments] = useState<DocumentMetadata[]>([]);
  const [loadingDocs, setLoadingDocs] = useState<boolean>(true);

  // Upload state
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState<boolean>(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [uploadSuccess, setUploadSuccess] = useState<string | null>(null);

  // Detail view state
  const [selectedDocId, setSelectedDocId] = useState<string | null>(null);
  const [docDetail, setDocDetail] = useState<DocumentDetail | null>(null);
  const [loadingDetail, setLoadingDetail] = useState<boolean>(false);
  const [detailError, setDetailError] = useState<string | null>(null);

  // Chunks state
  const [chunks, setChunks] = useState<DocumentChunk[]>([]);
  const [loadingChunks, setLoadingChunks] = useState<boolean>(false);
  const [chunksError, setChunksError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<"chunks" | "pages">("chunks");

  // Search state (Semantic vs Hybrid)
  const [searchMode, setSearchMode] = useState<SearchMode>("hybrid");
  const [executedMode, setExecutedMode] = useState<SearchMode>("hybrid");
  const [searchQuery, setSearchQuery] = useState<string>("");
  const [searchLimit, setSearchLimit] = useState<number>(5);
  const [searchResults, setSearchResults] = useState<SearchResultItem[] | null>(null);
  const [searching, setSearching] = useState<boolean>(false);
  const [searchError, setSearchError] = useState<string | null>(null);
  const [executedQuery, setExecutedQuery] = useState<string>("");

  // Document Indexing state
  const [indexingMap, setIndexingMap] = useState<Record<string, IndexState>>({});

  // Check backend health
  const checkHealth = () => {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 5000);

    fetch(`${API_URL}/health`, { signal: controller.signal })
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json();
      })
      .then((data: HealthData) => {
        setHealthData(data);
        setHealthStatus("healthy");
        setHealthError(null);
      })
      .catch((err) => {
        setHealthStatus("unavailable");
        setHealthError(
          err instanceof DOMException && err.name === "AbortError"
            ? "Request timed out"
            : err instanceof Error
              ? err.message
              : "Unknown error"
        );
      })
      .finally(() => clearTimeout(timeout));
  };

  // Fetch document listing
  const fetchDocuments = () => {
    fetch(`${API_URL}/api/documents`)
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json();
      })
      .then((data: { documents: DocumentMetadata[]; total: number }) => {
        setDocuments(data.documents);
      })
      .catch(() => {
        // Silently fail if backend is down; health card already alerts user
      })
      .finally(() => {
        setLoadingDocs(false);
      });
  };

  const handleRefresh = () => {
    setLoadingDocs(true);
    fetchDocuments();
  };

  useEffect(() => {
    checkHealth();
    fetchDocuments();
  }, []);

  // Handle PDF file selection
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0] || null;
    setUploadError(null);
    setUploadSuccess(null);

    if (file && !file.name.toLowerCase().endsWith(".pdf")) {
      setUploadError("Selected file must be a PDF document (.pdf).");
      setSelectedFile(null);
      return;
    }

    setSelectedFile(file);
  };

  // Handle form upload
  const handleUpload = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!selectedFile || uploading) return;

    setUploading(true);
    setUploadError(null);
    setUploadSuccess(null);

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      const res = await fetch(`${API_URL}/api/documents/upload`, {
        method: "POST",
        body: formData,
      });

      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.detail || `Upload failed with HTTP ${res.status}`);
      }

      setUploadSuccess(
        `Document "${data.filename}" successfully ingested: ${data.total_pages} page(s), ${data.total_characters} characters extracted and chunked.`
      );
      setSelectedFile(null);

      // Reset file input element
      const fileInput = document.getElementById("pdf-file-input") as HTMLInputElement;
      if (fileInput) fileInput.value = "";

      // Refresh list and auto-open detail view
      setLoadingDocs(true);
      fetchDocuments();
      openDetail(data.document_id);
    } catch (err) {
      setUploadError(err instanceof Error ? err.message : "Failed to upload document.");
    } finally {
      setUploading(false);
    }
  };

  // Open detail view for a document
  const openDetail = (docId: string) => {
    setSelectedDocId(docId);
    setLoadingDetail(true);
    setDetailError(null);
    setLoadingChunks(true);
    setChunksError(null);
    setActiveTab("chunks");

    // Fetch document metadata and pages
    fetch(`${API_URL}/api/documents/${docId}`)
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json();
      })
      .then((data: DocumentDetail) => {
        setDocDetail(data);
      })
      .catch((err) => {
        setDetailError(err instanceof Error ? err.message : "Failed to load document details.");
      })
      .finally(() => {
        setLoadingDetail(false);
      });

    // Fetch document chunks
    fetch(`${API_URL}/api/documents/${docId}/chunks`)
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json();
      })
      .then((data: { chunks: DocumentChunk[]; total_chunks: number }) => {
        setChunks(data.chunks);
      })
      .catch((err) => {
        setChunksError(err instanceof Error ? err.message : "Failed to load document chunks.");
      })
      .finally(() => {
        setLoadingChunks(false);
      });
  };

  const closeDetail = () => {
    setSelectedDocId(null);
    setDocDetail(null);
    setDetailError(null);
    setChunks([]);
    setChunksError(null);
  };

  // Handle document indexing for semantic search
  const handleIndexDocument = async (docId: string) => {
    setIndexingMap((prev) => ({
      ...prev,
      [docId]: { status: "indexing", message: "Generating embeddings & indexing..." },
    }));

    try {
      const res = await fetch(`${API_URL}/api/documents/${docId}/index`, {
        method: "POST",
      });

      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.detail || `Indexing failed with HTTP ${res.status}`);
      }

      setIndexingMap((prev) => ({
        ...prev,
        [docId]: {
          status: data.status === "empty" ? "empty" : "indexed",
          message: data.message,
        },
      }));
    } catch (err) {
      setIndexingMap((prev) => ({
        ...prev,
        [docId]: {
          status: "failed",
          message: err instanceof Error ? err.message : "Indexing failed",
        },
      }));
    }
  };

  // Handle search submission (Semantic vs Hybrid)
  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault();
    const trimmed = searchQuery.trim();
    if (!trimmed || searching) return;

    setSearching(true);
    setSearchError(null);
    setExecutedQuery(trimmed);
    setExecutedMode(searchMode);

    try {
      const endpoint = searchMode === "hybrid" ? "/api/search/hybrid" : "/api/search";
      const res = await fetch(`${API_URL}${endpoint}`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          query: trimmed,
          limit: searchLimit,
        }),
      });

      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.detail || `Search failed with HTTP ${res.status}`);
      }

      setSearchResults(data.results || []);
    } catch (err) {
      setSearchError(
        err instanceof Error
          ? err.message
          : `Failed to execute ${searchMode} search.`
      );
      setSearchResults(null);
    } finally {
      setSearching(false);
    }
  };

  return (
    <main className="min-h-screen bg-gray-50 text-gray-900 dark:bg-gray-950 dark:text-gray-100 py-8 px-4 sm:px-6 lg:px-8">
      <div className="max-w-5xl mx-auto space-y-8">
        {/* Header */}
        <header className="border-b border-gray-200 dark:border-gray-800 pb-5">
          <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
            <div>
              <h1 className="text-2xl font-bold tracking-tight">PolicyPilot</h1>
              <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
                AI-powered company policy assistant — Phase 5: BM25 & Hybrid Search with RRF
              </p>
            </div>

            {/* Backend health status badge */}
            <div className="flex items-center gap-2 border border-gray-200 dark:border-gray-800 rounded-lg px-3 py-1.5 bg-white dark:bg-gray-900 text-xs">
              <span className="text-gray-500 dark:text-gray-400 font-medium">Backend:</span>
              {healthStatus === "loading" && (
                <span className="flex items-center gap-1.5 text-yellow-600 dark:text-yellow-400">
                  <span className="h-2 w-2 rounded-full bg-yellow-400 animate-pulse"></span>
                  Connecting…
                </span>
              )}
              {healthStatus === "healthy" && healthData && (
                <span className="flex items-center gap-1.5 text-green-600 dark:text-green-400 font-medium">
                  <span className="h-2 w-2 rounded-full bg-green-500"></span>
                  Healthy (v{healthData.version})
                </span>
              )}
              {healthStatus === "unavailable" && (
                <span className="flex items-center gap-1.5 text-red-600 dark:text-red-400 font-medium">
                  <span className="h-2 w-2 rounded-full bg-red-500"></span>
                  Unavailable
                </span>
              )}
            </div>
          </div>
          {healthStatus === "unavailable" && (
            <p className="text-xs text-red-600 dark:text-red-400 mt-2">
              Could not reach {API_URL}/health: {healthError}
            </p>
          )}
        </header>

        {/* Search Section (Hybrid & Semantic) */}
        <section
          aria-labelledby="search-heading"
          className="bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-800 rounded-lg p-6 shadow-sm space-y-4"
        >
          <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
            <div>
              <h2 id="search-heading" className="text-base font-semibold text-gray-900 dark:text-gray-100 flex items-center gap-2">
                <span>Policy Retrieval</span>
                <span className="text-[11px] font-normal px-2 py-0.5 rounded bg-emerald-100 text-emerald-800 dark:bg-emerald-900/40 dark:text-emerald-300">
                  BM25 + RRF
                </span>
                <span className="text-[11px] font-normal px-2 py-0.5 rounded bg-blue-100 text-blue-800 dark:bg-blue-900/40 dark:text-blue-300">
                  Qdrant Vector
                </span>
              </h2>
              <p className="text-xs text-gray-500 dark:text-gray-400 mt-0.5">
                Query internal policies with semantic vector search, BM25 keyword matching, or hybrid fusion.
              </p>
            </div>

            {/* Search Mode Toggle */}
            <div className="flex items-center gap-2">
              <span className="text-xs text-gray-500 dark:text-gray-400 font-medium">Mode:</span>
              <div className="inline-flex p-0.5 bg-gray-100 dark:bg-gray-800 rounded-md text-xs border border-gray-200 dark:border-gray-700">
                <button
                  id="search-mode-hybrid-button"
                  type="button"
                  onClick={() => setSearchMode("hybrid")}
                  className={`px-3 py-1 rounded transition-colors ${
                    searchMode === "hybrid"
                      ? "bg-white dark:bg-gray-900 text-blue-600 dark:text-blue-400 font-medium shadow-xs"
                      : "text-gray-500 hover:text-gray-900 dark:hover:text-gray-200"
                  }`}
                >
                  Hybrid (BM25 + Vector)
                </button>
                <button
                  id="search-mode-semantic-button"
                  type="button"
                  onClick={() => setSearchMode("semantic")}
                  className={`px-3 py-1 rounded transition-colors ${
                    searchMode === "semantic"
                      ? "bg-white dark:bg-gray-900 text-blue-600 dark:text-blue-400 font-medium shadow-xs"
                      : "text-gray-500 hover:text-gray-900 dark:hover:text-gray-200"
                  }`}
                >
                  Semantic (Vector)
                </button>
              </div>
            </div>
          </div>

          <div className="text-[11px] bg-blue-50/70 dark:bg-blue-950/30 border border-blue-200/80 dark:border-blue-900/40 rounded p-2.5 text-blue-800 dark:text-blue-300">
            ℹ️ <strong>Scope Note:</strong>{" "}
            {searchMode === "hybrid"
              ? "Hybrid mode combines Okapi BM25 keyword matching with dense Qdrant vector retrieval using Reciprocal Rank Fusion (RRF constant k=60)."
              : "Semantic mode executes dense vector retrieval with cosine similarity over FastEmbed embeddings stored in Qdrant."}
          </div>

          {/* Search Form */}
          <form onSubmit={handleSearch} className="space-y-3">
            <div className="flex flex-col sm:flex-row items-stretch sm:items-center gap-3">
              <input
                id="search-query-input"
                type="text"
                placeholder={
                  searchMode === "hybrid"
                    ? "Ask in natural language or enter exact policy codes (e.g., HR-2024, annual leave, business travel)..."
                    : "Ask or search policies (e.g., annual leave allowance, business travel expenses, health insurance)..."
                }
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                disabled={searching}
                className="flex-1 text-xs px-3.5 py-2.5 border border-gray-300 dark:border-gray-700 bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50"
              />
              <div className="flex items-center gap-2">
                <select
                  id="search-limit-select"
                  value={searchLimit}
                  onChange={(e) => setSearchLimit(Number(e.target.value))}
                  disabled={searching}
                  className="text-xs px-2.5 py-2.5 border border-gray-300 dark:border-gray-700 bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                  <option value={3}>Top 3</option>
                  <option value={5}>Top 5</option>
                  <option value={10}>Top 10</option>
                </select>
                <button
                  id="search-submit-button"
                  type="submit"
                  disabled={!searchQuery.trim() || searching}
                  className="px-5 py-2.5 text-xs font-medium text-white bg-blue-600 hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed rounded-md whitespace-nowrap transition-colors flex items-center gap-2"
                >
                  {searching ? (
                    <>
                      <span className="h-3 w-3 border-2 border-white border-t-transparent rounded-full animate-spin"></span>
                      Searching…
                    </>
                  ) : searchMode === "hybrid" ? (
                    "Search Hybrid"
                  ) : (
                    "Search Vectors"
                  )}
                </button>
              </div>
            </div>

            {searchError && (
              <div id="search-error-message" className="text-xs text-red-600 dark:text-red-400 bg-red-50 dark:bg-red-950/40 border border-red-200 dark:border-red-900/50 p-3 rounded">
                <strong>Search Error:</strong> {searchError}
              </div>
            )}
          </form>

          {/* Search Results Display */}
          {searchResults !== null && (
            <div className="mt-4 pt-4 border-t border-gray-200 dark:border-gray-800 space-y-3" id="search-results-container">
              <div className="flex items-center justify-between text-xs text-gray-600 dark:text-gray-400">
                <span>
                  Query: &ldquo;<strong className="text-gray-900 dark:text-gray-100">{executedQuery}</strong>&rdquo;
                  {" "}(<span className="capitalize font-medium text-gray-800 dark:text-gray-200">{executedMode}</span> mode) — Found{" "}
                  <strong>{searchResults.length}</strong> matching passage{searchResults.length === 1 ? "" : "s"}
                </span>
                <button
                  onClick={() => {
                    setSearchResults(null);
                    setSearchQuery("");
                  }}
                  className="text-gray-400 hover:text-gray-600 dark:hover:text-gray-200 underline"
                >
                  Clear results
                </button>
              </div>

              {searchResults.length === 0 ? (
                <div className="text-center py-6 border border-dashed border-gray-200 dark:border-gray-800 rounded-md">
                  <p className="text-xs text-gray-500 dark:text-gray-400">
                    No matching passages found. Ensure your documents have been uploaded and indexed using the &ldquo;Index for Search&rdquo; action below.
                  </p>
                </div>
              ) : (
                <div className="space-y-3">
                  {searchResults.map((result, idx) => (
                    <div
                      key={result.chunk_id}
                      className="border border-gray-200 dark:border-gray-800 rounded-lg overflow-hidden bg-white dark:bg-gray-900/80 shadow-xs"
                    >
                      {/* Result Header */}
                      <div className="bg-gray-100/70 dark:bg-gray-800/60 px-3.5 py-2 flex flex-wrap items-center justify-between gap-2 text-xs">
                        <div className="flex items-center gap-2">
                          <span className="font-semibold text-gray-800 dark:text-gray-100">
                            #{idx + 1}
                          </span>
                          <span className="font-medium text-gray-900 dark:text-gray-100 max-w-xs truncate">
                            {result.filename}
                          </span>
                          {result.section && (
                            <span className="inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-medium bg-blue-100 text-blue-800 dark:bg-blue-900/40 dark:text-blue-300">
                              {result.section}
                            </span>
                          )}
                        </div>
                        <div className="flex items-center gap-2.5 text-[11px]">
                          {executedMode === "hybrid" ? (
                            <>
                              {result.rrf_score !== undefined && (
                                <span className="inline-flex items-center px-1.5 py-0.5 rounded font-mono font-medium bg-emerald-100 text-emerald-800 dark:bg-emerald-900/40 dark:text-emerald-300" title="Reciprocal Rank Fusion score">
                                  RRF: {result.rrf_score.toFixed(6)}
                                </span>
                              )}
                              {result.vector_rank !== null && result.vector_rank !== undefined && (
                                <span className="inline-flex items-center px-1.5 py-0.5 rounded font-mono text-[10px] bg-blue-100 text-blue-800 dark:bg-blue-900/40 dark:text-blue-300">
                                  Vector #{result.vector_rank}
                                </span>
                              )}
                              {result.keyword_rank !== null && result.keyword_rank !== undefined && (
                                <span className="inline-flex items-center px-1.5 py-0.5 rounded font-mono text-[10px] bg-amber-100 text-amber-800 dark:bg-amber-900/40 dark:text-amber-300">
                                  BM25 #{result.keyword_rank}
                                </span>
                              )}
                            </>
                          ) : (
                            <>
                              {result.score !== undefined && (
                                <span className="inline-flex items-center px-1.5 py-0.5 rounded font-mono font-medium bg-green-100 text-green-800 dark:bg-green-900/40 dark:text-green-300">
                                  Cosine: {result.score.toFixed(4)}
                                </span>
                              )}
                              <span className="inline-flex items-center px-1.5 py-0.5 rounded font-mono text-[10px] bg-blue-100 text-blue-800 dark:bg-blue-900/40 dark:text-blue-300">
                                Vector
                              </span>
                            </>
                          )}
                          <span className="text-gray-500 dark:text-gray-400">
                            Page{result.page_numbers.length > 1 ? "s" : ""}:{" "}
                            <strong className="text-gray-700 dark:text-gray-300 font-mono">
                              {result.page_numbers.join(", ")}
                            </strong>
                          </span>
                          <button
                            onClick={() => openDetail(result.document_id)}
                            className="text-blue-600 hover:text-blue-800 dark:text-blue-400 dark:hover:text-blue-300 font-medium underline"
                          >
                            Inspect Document
                          </button>
                        </div>
                      </div>

                      {/* Result Content */}
                      <div className="p-3.5">
                        <pre className="text-xs whitespace-pre-wrap font-mono text-gray-800 dark:text-gray-200 leading-relaxed bg-gray-50 dark:bg-gray-950/60 p-3 rounded border border-gray-100 dark:border-gray-800/80">
                          {result.text}
                        </pre>
                        <div className="mt-2 flex items-center justify-between text-[10px] text-gray-400 dark:text-gray-500 font-mono">
                          <span>Chunk #{result.chunk_index} ({result.character_count} chars)</span>
                          <span>ID: {result.chunk_id}</span>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </section>

        {/* Upload Section */}
        <section aria-labelledby="upload-heading" className="bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-800 rounded-lg p-6 shadow-sm">
          <h2 id="upload-heading" className="text-base font-semibold text-gray-900 dark:text-gray-100 mb-2">
            Upload Policy PDF
          </h2>
          <p className="text-xs text-gray-500 dark:text-gray-400 mb-4">
            Upload a policy document in PDF format. Text is extracted page-by-page and partitioned into structured chunks ready for indexing.
          </p>

          <form onSubmit={handleUpload} className="space-y-4">
            <div className="flex flex-col sm:flex-row items-stretch sm:items-center gap-3">
              <input
                id="pdf-file-input"
                type="file"
                accept=".pdf,application/pdf"
                onChange={handleFileChange}
                disabled={uploading}
                className="block w-full text-xs text-gray-700 dark:text-gray-300 file:mr-4 file:py-2 file:px-4 file:rounded file:border-0 file:text-xs file:font-semibold file:bg-gray-100 file:text-gray-800 hover:file:bg-gray-200 dark:file:bg-gray-800 dark:file:text-gray-200 dark:hover:file:bg-gray-700 cursor-pointer border border-gray-300 dark:border-gray-700 rounded-md p-1"
              />
              <button
                id="upload-button"
                type="submit"
                disabled={!selectedFile || uploading}
                className="px-5 py-2 text-xs font-medium text-white bg-blue-600 hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed rounded-md whitespace-nowrap transition-colors"
              >
                {uploading ? "Ingesting & Chunking…" : "Upload & Ingest"}
              </button>
            </div>

            {/* Error & Success Feedback */}
            {uploadError && (
              <div id="upload-error-message" className="text-xs text-red-600 dark:text-red-400 bg-red-50 dark:bg-red-950/40 border border-red-200 dark:border-red-900/50 p-3 rounded">
                <strong>Upload Error:</strong> {uploadError}
              </div>
            )}
            {uploadSuccess && (
              <div id="upload-success-message" className="text-xs text-green-700 dark:text-green-300 bg-green-50 dark:bg-green-950/40 border border-green-200 dark:border-green-900/50 p-3 rounded">
                <strong>Success:</strong> {uploadSuccess}
              </div>
            )}
          </form>
        </section>

        {/* Document Library Section */}
        <section aria-labelledby="documents-heading" className="bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-800 rounded-lg p-6 shadow-sm">
          <div className="flex items-center justify-between mb-4">
            <h2 id="documents-heading" className="text-base font-semibold text-gray-900 dark:text-gray-100">
              Ingested Document Library ({documents.length})
            </h2>
            <button
              onClick={handleRefresh}
              disabled={loadingDocs}
              className="text-xs text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200 underline"
            >
              Refresh list
            </button>
          </div>

          {loadingDocs ? (
            <p className="text-xs text-gray-500 dark:text-gray-400 py-6 text-center">
              Loading document library…
            </p>
          ) : documents.length === 0 ? (
            <div className="text-center py-8 border border-dashed border-gray-200 dark:border-gray-800 rounded-lg">
              <p className="text-xs text-gray-500 dark:text-gray-400">
                No policy documents have been uploaded yet. Upload a PDF above to begin.
              </p>
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs border-collapse">
                <thead>
                  <tr className="border-b border-gray-200 dark:border-gray-800 text-gray-500 dark:text-gray-400 font-medium">
                    <th className="py-2.5 px-3">Filename</th>
                    <th className="py-2.5 px-3">Pages</th>
                    <th className="py-2.5 px-3">Characters</th>
                    <th className="py-2.5 px-3">Size</th>
                    <th className="py-2.5 px-3">Status</th>
                    <th className="py-2.5 px-3">Vector Index</th>
                    <th className="py-2.5 px-3 text-right">Actions</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-100 dark:divide-gray-800/60">
                  {documents.map((doc) => {
                    const idxState = indexingMap[doc.document_id];
                    return (
                      <tr key={doc.document_id} className="hover:bg-gray-50/70 dark:hover:bg-gray-800/40">
                        <td className="py-2.5 px-3 font-medium text-gray-900 dark:text-gray-100 max-w-xs truncate">
                          {doc.filename}
                        </td>
                        <td className="py-2.5 px-3">{doc.total_pages}</td>
                        <td className="py-2.5 px-3">{doc.total_characters.toLocaleString()}</td>
                        <td className="py-2.5 px-3">{(doc.file_size_bytes / 1024).toFixed(1)} KB</td>
                        <td className="py-2.5 px-3">
                          <span className="inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-medium bg-green-100 text-green-800 dark:bg-green-900/40 dark:text-green-300">
                            {doc.ingestion_status}
                          </span>
                        </td>
                        <td className="py-2.5 px-3 whitespace-nowrap">
                          {idxState?.status === "indexing" ? (
                            <span className="inline-flex items-center gap-1 text-[11px] text-yellow-600 dark:text-yellow-400 font-medium">
                              <span className="h-2 w-2 rounded-full bg-yellow-400 animate-pulse"></span>
                              Indexing…
                            </span>
                          ) : idxState?.status === "indexed" ? (
                            <span className="inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-medium bg-purple-100 text-purple-800 dark:bg-purple-900/40 dark:text-purple-300">
                              ✓ Indexed
                            </span>
                          ) : idxState?.status === "empty" ? (
                            <span className="inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-medium bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-gray-400">
                              No Chunks
                            </span>
                          ) : idxState?.status === "failed" ? (
                            <span className="inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-medium bg-red-100 text-red-800 dark:bg-red-900/40 dark:text-red-300" title={idxState.message}>
                              Error
                            </span>
                          ) : (
                            <button
                              onClick={() => handleIndexDocument(doc.document_id)}
                              className="inline-flex items-center gap-1 text-[11px] font-medium px-2 py-0.5 rounded bg-blue-50 text-blue-700 hover:bg-blue-100 dark:bg-blue-900/30 dark:text-blue-300 dark:hover:bg-blue-900/50 transition-colors"
                            >
                              ⚡ Index
                            </button>
                          )}
                        </td>
                        <td className="py-2.5 px-3 text-right space-x-2 whitespace-nowrap">
                          {idxState?.status === "indexed" && (
                            <button
                              onClick={() => handleIndexDocument(doc.document_id)}
                              className="text-xs text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200"
                              title="Reindex document vectors"
                            >
                              Reindex
                            </button>
                          )}
                          <button
                            onClick={() => openDetail(doc.document_id)}
                            className="text-xs text-blue-600 hover:text-blue-800 dark:text-blue-400 dark:hover:text-blue-300 font-medium"
                          >
                            View Details
                          </button>
                          <span className="text-gray-300 dark:text-gray-700">|</span>
                          <a
                            href={`${API_URL}/api/documents/${doc.document_id}/file`}
                            target="_blank"
                            rel="noreferrer"
                            className="text-xs text-gray-600 hover:text-gray-900 dark:text-gray-400 dark:hover:text-gray-200"
                          >
                            PDF
                          </a>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
        </section>

        {/* Document Detail & Chunks View */}
        {selectedDocId && (
          <section
            aria-labelledby="detail-heading"
            id="document-detail-panel"
            className="bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-800 rounded-lg p-6 shadow-sm space-y-5"
          >
            {/* Header */}
            <div className="flex items-start justify-between border-b border-gray-200 dark:border-gray-800 pb-3">
              <div>
                <h3 id="detail-heading" className="text-base font-semibold text-gray-900 dark:text-gray-100">
                  {docDetail?.filename ?? "Loading document…"}
                </h3>
                {docDetail && (
                  <p className="text-xs text-gray-500 dark:text-gray-400 mt-0.5">
                    ID: <code className="font-mono">{docDetail.document_id}</code> | SHA-256:{" "}
                    <code className="font-mono text-[11px]">{docDetail.sha256_hash.slice(0, 16)}…</code>
                  </p>
                )}
              </div>
              <div className="flex items-center gap-2">
                {docDetail && (
                  <button
                    onClick={() => handleIndexDocument(docDetail.document_id)}
                    disabled={indexingMap[docDetail.document_id]?.status === "indexing"}
                    className="text-xs px-3 py-1 bg-blue-50 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300 hover:bg-blue-100 dark:hover:bg-blue-900/50 rounded font-medium disabled:opacity-50"
                  >
                    {indexingMap[docDetail.document_id]?.status === "indexing"
                      ? "Indexing…"
                      : indexingMap[docDetail.document_id]?.status === "indexed"
                        ? "✓ Reindex in Qdrant"
                        : "⚡ Index in Qdrant"}
                  </button>
                )}
                {docDetail && (
                  <a
                    href={`${API_URL}/api/documents/${docDetail.document_id}/file`}
                    target="_blank"
                    rel="noreferrer"
                    className="text-xs px-3 py-1 bg-gray-100 dark:bg-gray-800 hover:bg-gray-200 dark:hover:bg-gray-700 rounded text-gray-700 dark:text-gray-300 font-medium"
                  >
                    Open PDF in Viewer
                  </a>
                )}
                <button
                  onClick={closeDetail}
                  className="text-xs text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200 px-2 py-1 rounded"
                >
                  Close
                </button>
              </div>
            </div>

            {loadingDetail ? (
              <p className="text-xs text-gray-500 dark:text-gray-400 py-6 text-center">
                Fetching document details…
              </p>
            ) : detailError ? (
              <div className="text-xs text-red-600 dark:text-red-400 p-3 bg-red-50 dark:bg-red-950/40 rounded">
                {detailError}
              </div>
            ) : docDetail ? (
              <div className="space-y-4">
                {/* Stats summary bar */}
                <div className="flex flex-wrap items-center gap-4 text-xs text-gray-600 dark:text-gray-400 bg-gray-50 dark:bg-gray-800/40 p-3 rounded">
                  <span>
                    Total Pages: <strong className="text-gray-900 dark:text-gray-100">{docDetail.total_pages}</strong>
                  </span>
                  <span>
                    Total Characters: <strong className="text-gray-900 dark:text-gray-100">{docDetail.total_characters.toLocaleString()}</strong>
                  </span>
                  <span>
                    Total Chunks: <strong className="text-gray-900 dark:text-gray-100">{chunks.length}</strong>
                  </span>
                  <span>
                    Status: <strong className="text-gray-900 dark:text-gray-100">{docDetail.ingestion_status}</strong>
                  </span>
                </div>

                {/* Tabs */}
                <div className="border-b border-gray-200 dark:border-gray-800 flex gap-4 text-xs">
                  <button
                    id="tab-chunks"
                    onClick={() => setActiveTab("chunks")}
                    className={`pb-2 font-medium transition-colors border-b-2 ${
                      activeTab === "chunks"
                        ? "border-blue-600 text-blue-600 dark:text-blue-400"
                        : "border-transparent text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200"
                    }`}
                  >
                    Structured Chunks ({chunks.length})
                  </button>
                  <button
                    id="tab-pages"
                    onClick={() => setActiveTab("pages")}
                    className={`pb-2 font-medium transition-colors border-b-2 ${
                      activeTab === "pages"
                        ? "border-blue-600 text-blue-600 dark:text-blue-400"
                        : "border-transparent text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200"
                    }`}
                  >
                    Raw Extracted Pages ({docDetail.total_pages})
                  </button>
                </div>

                {/* Chunks View */}
                {activeTab === "chunks" && (
                  <div className="space-y-4" id="chunks-container">
                    {loadingChunks ? (
                      <p className="text-xs text-gray-500 dark:text-gray-400 py-6 text-center">
                        Loading document chunks…
                      </p>
                    ) : chunksError ? (
                      <div className="text-xs text-red-600 dark:text-red-400 p-3 bg-red-50 dark:bg-red-950/40 rounded">
                        {chunksError}
                      </div>
                    ) : chunks.length === 0 ? (
                      <p className="text-xs italic text-gray-500 dark:text-gray-400 py-4 text-center">
                        No chunks available for this document. (Document has no extractable text).
                      </p>
                    ) : (
                      chunks.map((chunk) => (
                        <div
                          key={chunk.chunk_id}
                          className="border border-gray-200 dark:border-gray-800 rounded-md overflow-hidden bg-white dark:bg-gray-900"
                        >
                          <div className="bg-gray-100/70 dark:bg-gray-800/60 px-3 py-2 flex flex-wrap items-center justify-between gap-2 text-xs">
                            <div className="flex items-center gap-2">
                              <span className="font-semibold text-gray-800 dark:text-gray-100">
                                Chunk #{chunk.chunk_index}
                              </span>
                              {chunk.section && (
                                <span className="inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-medium bg-blue-100 text-blue-800 dark:bg-blue-900/40 dark:text-blue-300">
                                  {chunk.section}
                                </span>
                              )}
                            </div>
                            <div className="flex items-center gap-3 text-[11px] text-gray-500 dark:text-gray-400">
                              <span>
                                Page{chunk.page_numbers.length > 1 ? "s" : ""}:{" "}
                                <strong className="text-gray-700 dark:text-gray-300 font-mono">
                                  {chunk.page_numbers.join(", ")}
                                </strong>
                              </span>
                              <span>•</span>
                              <span>{chunk.character_count.toLocaleString()} chars</span>
                            </div>
                          </div>
                          <div className="p-3">
                            <pre className="text-xs whitespace-pre-wrap font-mono text-gray-800 dark:text-gray-200 leading-relaxed max-h-64 overflow-y-auto bg-gray-50 dark:bg-gray-950/60 p-2.5 rounded border border-gray-100 dark:border-gray-800/80">
                              {chunk.text}
                            </pre>
                            <p className="text-[10px] text-gray-400 dark:text-gray-600 mt-1 font-mono">
                              Chunk ID: {chunk.chunk_id}
                            </p>
                          </div>
                        </div>
                      ))
                    )}
                  </div>
                )}

                {/* Raw Pages View */}
                {activeTab === "pages" && (
                  <div className="space-y-4" id="pages-container">
                    {docDetail.pages.map((p) => (
                      <div
                        key={p.page_number}
                        className="border border-gray-200 dark:border-gray-800 rounded-md overflow-hidden"
                      >
                        <div className="bg-gray-100/70 dark:bg-gray-800/60 px-3 py-1.5 flex items-center justify-between text-xs">
                          <span className="font-semibold text-gray-700 dark:text-gray-200">
                            Page {p.page_number} of {docDetail.total_pages}
                          </span>
                          <span className="text-gray-500 dark:text-gray-400 text-[11px]">
                            {p.character_count.toLocaleString()} characters
                          </span>
                        </div>
                        <div className="p-3 bg-white dark:bg-gray-900">
                          {p.text.trim().length > 0 ? (
                            <pre className="text-xs whitespace-pre-wrap font-mono text-gray-800 dark:text-gray-200 leading-relaxed max-h-72 overflow-y-auto">
                              {p.text}
                            </pre>
                          ) : (
                            <p className="text-xs italic text-gray-400 dark:text-gray-500 py-2">
                              (No extractable text found on this page. If this page is an image or scanned document, OCR is not supported in Phase 2/3.)
                            </p>
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            ) : null}
          </section>
        )}
      </div>
    </main>
  );
}
