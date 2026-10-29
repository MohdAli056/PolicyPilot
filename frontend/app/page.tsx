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

type SearchMode = "rerank" | "hybrid" | "semantic";

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
  rerank_score?: number;
  rerank_rank?: number;
}

interface IndexState {
  status: "idle" | "indexing" | "indexed" | "empty" | "failed";
  message?: string;
}

interface CitationItem {
  source_id: string;
  document_id: string;
  filename: string;
  chunk_id: string;
  page_numbers: number[];
  section: string | null;
  excerpt: string;
}

interface SourceItem {
  source_id: string;
  document_id: string;
  filename: string;
  chunk_id: string;
  chunk_index: number;
  page_numbers: number[];
  section: string | null;
  text: string;
  rerank_score: number;
  rerank_rank: number;
  rrf_score: number;
  retrievers: string[];
}

interface AskResponseData {
  question: string;
  answer: string;
  is_abstention: boolean;
  citations: CitationItem[];
  sources: SourceItem[];
  model: string;
  timing_ms?: {
    retrieval_ms?: number;
    rerank_ms?: number;
    llm_ms?: number;
    total_ms?: number;
  };
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

  // Search state (Semantic, Hybrid, Reranked)
  const [searchMode, setSearchMode] = useState<SearchMode>("rerank");
  const [executedMode, setExecutedMode] = useState<SearchMode>("rerank");
  const [searchQuery, setSearchQuery] = useState<string>("");
  const [searchLimit, setSearchLimit] = useState<number>(5);
  const [searchResults, setSearchResults] = useState<SearchResultItem[] | null>(null);
  const [searching, setSearching] = useState<boolean>(false);
  const [searchError, setSearchError] = useState<string | null>(null);
  const [executedQuery, setExecutedQuery] = useState<string>("");

  // Question Answering (RAG) state
  const [qaQuestion, setQaQuestion] = useState<string>("");
  const [qaLimit, setQaLimit] = useState<number>(5);
  const [asking, setAsking] = useState<boolean>(false);
  const [qaError, setQaError] = useState<string | null>(null);
  const [qaResponse, setQaResponse] = useState<AskResponseData | null>(null);
  const [selectedCitationId, setSelectedCitationId] = useState<string | null>(null);
  const [showAllSources, setShowAllSources] = useState<boolean>(false);

  // Document Indexing and Deletion state
  const [indexingMap, setIndexingMap] = useState<Record<string, IndexState>>({});
  const [deletingDocId, setDeletingDocId] = useState<string | null>(null);

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

  // Handle document deletion
  const handleDeleteDocument = async (docId: string, filename: string) => {
    const confirmed = window.confirm(
      `Are you sure you want to permanently delete "${filename}"?\n\nThis will remove the uploaded PDF, all extracted text chunks, BM25 keyword index, and Qdrant vector representations.`
    );
    if (!confirmed) return;

    setDeletingDocId(docId);
    try {
      const res = await fetch(`${API_URL}/api/documents/${docId}`, {
        method: "DELETE",
      });

      if (!res.ok) {
        const errorData = await res.json().catch(() => null);
        throw new Error(errorData?.detail || `Failed to delete document (${res.status})`);
      }

      // Remove from documents list
      setDocuments((prev) => prev.filter((d) => d.document_id !== docId));

      // Remove from indexing state
      setIndexingMap((prev) => {
        const next = { ...prev };
        delete next[docId];
        return next;
      });

      // Close detail view if this document was selected
      if (selectedDocId === docId) {
        closeDetail();
      }
    } catch (err) {
      alert(err instanceof Error ? err.message : "Failed to delete document.");
    } finally {
      setDeletingDocId(null);
    }
  };

  const handleModeChange = (mode: SearchMode) => {
    setSearchMode(mode);
    setSearchResults(null);
    setSearchError(null);
  };

  // Handle search submission (Semantic, Hybrid, or Reranked)
  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault();
    const trimmed = searchQuery.trim();
    if (!trimmed || searching) return;

    setSearching(true);
    setSearchError(null);
    setExecutedQuery(trimmed);
    setExecutedMode(searchMode);

    try {
      const endpoint =
        searchMode === "rerank"
          ? "/api/search/rerank"
          : searchMode === "hybrid"
            ? "/api/search/hybrid"
            : "/api/search";
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

  // Handle Question Answering submission
  const handleAsk = async (e: React.FormEvent) => {
    e.preventDefault();
    const trimmed = qaQuestion.trim();
    if (!trimmed || asking) return;

    setAsking(true);
    setQaError(null);
    setQaResponse(null);
    setSelectedCitationId(null);

    try {
      const res = await fetch(`${API_URL}/api/ask`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          question: trimmed,
          limit: qaLimit,
        }),
      });

      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.detail || `Question answering failed with HTTP ${res.status}`);
      }

      setQaResponse(data);
    } catch (err) {
      setQaError(err instanceof Error ? err.message : "Failed to answer question.");
      setQaResponse(null);
    } finally {
      setAsking(false);
    }
  };

  const handleClearQA = () => {
    setQaQuestion("");
    setQaResponse(null);
    setQaError(null);
    setSelectedCitationId(null);
  };

  // Safely renders answer text with clickable interactive citation pills
  const renderAnswerWithCitations = (
    answerText: string,
    onCitationClick: (sourceId: string) => void
  ) => {
    const parts = answerText.split(/(\[S\d+\])/g);
    return (
      <span>
        {parts.map((part, idx) => {
          const match = part.match(/^\[(S\d+)\]$/);
          if (match) {
            const sourceId = match[1];
            const isSelected = selectedCitationId === sourceId;
            return (
              <button
                key={idx}
                type="button"
                onClick={() => onCitationClick(sourceId)}
                className={`inline-flex items-center px-1.5 py-0.2 mx-0.5 rounded text-[11px] font-mono font-bold transition-all ${
                  isSelected
                    ? "bg-purple-600 text-white ring-2 ring-purple-400"
                    : "bg-blue-100 text-blue-800 dark:bg-blue-900/60 dark:text-blue-300 hover:bg-blue-200 dark:hover:bg-blue-800"
                }`}
                title={`View verified citation details for [${sourceId}]`}
              >
                [{sourceId}]
              </button>
            );
          }
          return <span key={idx}>{part}</span>;
        })}
      </span>
    );
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
                AI-powered enterprise policy assistant with verifiable citations & multi-stage retrieval
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

        {/* Grounded Question Answering Section */}
        <section
          aria-labelledby="qa-heading"
          className="bg-white dark:bg-gray-900 border border-blue-200/70 dark:border-blue-900/50 rounded-lg p-6 shadow-sm space-y-5"
        >
          <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
            <div>
              <h2
                id="qa-heading"
                className="text-base font-semibold text-gray-900 dark:text-gray-100 flex items-center gap-2"
              >
                <span>Ask PolicyPilot</span>
                <span className="text-[11px] font-normal px-2 py-0.5 rounded bg-blue-100 text-blue-800 dark:bg-blue-900/40 dark:text-blue-300">
                  Grounded RAG
                </span>
                <span className="text-[11px] font-normal px-2 py-0.5 rounded bg-purple-100 text-purple-800 dark:bg-purple-900/40 dark:text-purple-300">
                  Evidence Citations
                </span>
              </h2>
              <p className="text-xs text-gray-500 dark:text-gray-400 mt-0.5">
                Ask natural-language questions about company policies. Answers are synthesized strictly from retrieved evidence with verified page citations.
              </p>
            </div>
          </div>

          {/* Question Form */}
          <form onSubmit={handleAsk} className="space-y-3">
            <div className="flex flex-col sm:flex-row items-stretch sm:items-center gap-3">
              <input
                id="qa-question-input"
                type="text"
                placeholder="Ask a question about internal company policies (e.g., How many days of annual leave do I get?)..."
                value={qaQuestion}
                onChange={(e) => {
                  setQaQuestion(e.target.value);
                  if (qaResponse !== null) {
                    setQaResponse(null);
                  }
                }}
                disabled={asking}
                className="flex-1 text-xs px-3.5 py-2.5 border border-gray-300 dark:border-gray-700 bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50"
              />
              <div className="flex items-center gap-2">
                <select
                  id="qa-limit-select"
                  value={qaLimit}
                  onChange={(e) => setQaLimit(Number(e.target.value))}
                  disabled={asking}
                  className="text-xs px-2.5 py-2.5 border border-gray-300 dark:border-gray-700 bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                  <option value={3}>Top 3 Evidence</option>
                  <option value={5}>Top 5 Evidence</option>
                  <option value={10}>Top 10 Evidence</option>
                </select>
                <button
                  id="qa-submit-button"
                  type="submit"
                  disabled={!qaQuestion.trim() || asking}
                  className="px-5 py-2.5 text-xs font-medium text-white bg-blue-600 hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed rounded-md whitespace-nowrap transition-colors flex items-center gap-2"
                >
                  {asking ? (
                    <>
                      <span className="h-3 w-3 border-2 border-white border-t-transparent rounded-full animate-spin"></span>
                      Synthesizing…
                    </>
                  ) : (
                    "Ask PolicyPilot"
                  )}
                </button>
              </div>
            </div>

            {qaError && (
              <div
                id="qa-error-message"
                className="text-xs text-red-600 dark:text-red-400 bg-red-50 dark:bg-red-950/40 border border-red-200 dark:border-red-900/50 p-3 rounded"
              >
                <strong>Question Answering Error:</strong> {qaError}
              </div>
            )}
          </form>

          {/* QA Results Display */}
          {qaResponse !== null && (
            <div
              id="qa-results-container"
              className="mt-4 pt-4 border-t border-gray-200 dark:border-gray-800 space-y-4"
            >
              <div className="flex items-center justify-between text-xs text-gray-600 dark:text-gray-400">
                <span>
                  Question: &ldquo;<strong className="text-gray-900 dark:text-gray-100">{qaResponse.question}</strong>&rdquo;
                </span>
                <button
                  id="qa-clear-button"
                  onClick={handleClearQA}
                  className="text-gray-400 hover:text-gray-600 dark:hover:text-gray-200 underline"
                >
                  Clear answer
                </button>
              </div>

              {/* Abstention State */}
              {qaResponse.is_abstention ? (
                <div
                  id="qa-abstention-notice"
                  className="bg-amber-50 dark:bg-amber-950/30 border border-amber-200 dark:border-amber-900/50 rounded-lg p-4 space-y-2 text-xs"
                >
                  <div className="flex items-center gap-2 text-amber-800 dark:text-amber-300 font-semibold">
                    <span>⚠️</span>
                    <span>Insufficient Evidence in Uploaded Policies</span>
                  </div>
                  <p className="text-amber-900/90 dark:text-amber-200/90 leading-relaxed font-medium">
                    {qaResponse.answer}
                  </p>
                  <p className="text-[11px] text-amber-700/80 dark:text-amber-400/80">
                    The uploaded policy documents do not establish an answer to this question. Please ensure relevant policy PDFs are uploaded and indexed.
                  </p>
                </div>
              ) : (
                /* Grounded Answer Card */
                <div
                  id="qa-answer-card"
                  className="bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-800 rounded-lg p-5 shadow-xs space-y-4"
                >
                  <div className="flex items-center justify-between">
                    <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded text-xs font-semibold bg-emerald-100 text-emerald-800 dark:bg-emerald-950/60 dark:text-emerald-300">
                      <span className="h-1.5 w-1.5 rounded-full bg-emerald-500"></span>
                      Grounded Answer
                    </span>
                    <span className="text-[11px] text-gray-500 dark:text-gray-400 font-mono">
                      Model: {qaResponse.model}
                      {qaResponse.timing_ms?.total_ms !== undefined && (
                        <> • {qaResponse.timing_ms.total_ms}ms total</>
                      )}
                    </span>
                  </div>

                  {/* Answer Text with Interactive Citation Pills */}
                  <div
                    id="qa-answer-text"
                    className="text-sm leading-relaxed text-gray-800 dark:text-gray-200 border-l-2 border-emerald-500 pl-3 py-1"
                  >
                    {renderAnswerWithCitations(qaResponse.answer, (srcId) => {
                      setSelectedCitationId((prev) => (prev === srcId ? null : srcId));
                    })}
                  </div>

                  {/* Verified Citations List */}
                  {qaResponse.citations.length > 0 && (
                    <div id="qa-citations-section" className="space-y-2.5 pt-2">
                      <h3 className="text-xs font-semibold text-gray-900 dark:text-gray-100 flex items-center gap-2">
                        <span>Supporting Citations ({qaResponse.citations.length})</span>
                        <span className="text-[10px] font-normal text-gray-500">
                          (Click [S#] badges above to highlight citations)
                        </span>
                      </h3>
                      <div className="grid grid-cols-1 gap-2.5">
                        {qaResponse.citations.map((c) => {
                          const isHighlighted = selectedCitationId === c.source_id;
                          return (
                            <div
                              key={c.source_id}
                              id={`citation-${c.source_id}`}
                              className={`text-xs p-3 rounded-lg border transition-all ${
                                isHighlighted
                                  ? "border-purple-500 bg-purple-50/50 dark:bg-purple-950/30 ring-1 ring-purple-400"
                                  : "border-gray-200 dark:border-gray-800 bg-gray-50/60 dark:bg-gray-800/40"
                              }`}
                            >
                              <div className="flex flex-wrap items-center justify-between gap-2 pb-1.5 border-b border-gray-200/60 dark:border-gray-700/50">
                                <div className="flex items-center gap-2">
                                  <span className="px-1.5 py-0.5 rounded font-mono font-bold bg-blue-100 text-blue-800 dark:bg-blue-900/60 dark:text-blue-300 text-[11px]">
                                    [{c.source_id}]
                                  </span>
                                  <span className="font-medium text-gray-900 dark:text-gray-100">
                                    {c.filename}
                                  </span>
                                  <span className="text-[11px] text-gray-500 dark:text-gray-400">
                                    (Page{c.page_numbers.length > 1 ? "s" : ""} {c.page_numbers.join(", ")})
                                  </span>
                                  {c.section && (
                                    <span className="text-[10px] px-1.5 py-0.2 rounded bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300">
                                      {c.section}
                                    </span>
                                  )}
                                </div>
                                <div className="flex items-center gap-2">
                                  <button
                                    type="button"
                                    onClick={() => openDetail(c.document_id)}
                                    className="text-[11px] text-blue-600 hover:text-blue-700 dark:text-blue-400 underline font-medium"
                                  >
                                    Inspect Document
                                  </button>
                                  <a
                                    href={`${API_URL}/api/documents/${c.document_id}/file`}
                                    target="_blank"
                                    rel="noreferrer"
                                    className="text-[11px] text-gray-500 hover:text-gray-700 dark:text-gray-400 underline"
                                  >
                                    View PDF ↗
                                  </a>
                                </div>
                              </div>
                              <blockquote className="mt-2 text-[11px] text-gray-600 dark:text-gray-300 italic pl-2 border-l-2 border-gray-300 dark:border-gray-600">
                                &ldquo;{c.excerpt}&rdquo;
                              </blockquote>
                            </div>
                          );
                        })}
                      </div>
                    </div>
                  )}

                  {/* Expandable Retrieved Sources Context */}
                  {qaResponse.sources.length > 0 && (
                    <div className="pt-2">
                      <button
                        type="button"
                        onClick={() => setShowAllSources((prev) => !prev)}
                        className="text-[11px] text-gray-500 hover:text-gray-700 dark:text-gray-400 flex items-center gap-1 font-medium underline"
                      >
                        <span>{showAllSources ? "▼ Hide" : "▶ View"} All Supplied Evidence Passages ({qaResponse.sources.length})</span>
                      </button>
                      {showAllSources && (
                        <div className="mt-2 space-y-2 border border-gray-200 dark:border-gray-800 rounded-lg p-3 bg-gray-50/50 dark:bg-gray-900/50">
                          {qaResponse.sources.map((s) => (
                            <div
                              key={s.chunk_id}
                              className="text-[11px] border-b border-gray-200 dark:border-gray-800 pb-2 last:border-b-0"
                            >
                              <div className="flex items-center justify-between text-gray-600 dark:text-gray-400 font-mono text-[10px]">
                                <span>[{s.source_id}] {s.filename} (p. {s.page_numbers.join(", ")})</span>
                                <span>Rerank Score: {s.rerank_score.toFixed(6)} | RRF: {s.rrf_score.toFixed(6)}</span>
                              </div>
                              <p className="text-gray-800 dark:text-gray-200 mt-1 line-clamp-3">
                                {s.text}
                              </p>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  )}
                </div>
              )}
            </div>
          )}
        </section>

        {/* Search Section (Reranked, Hybrid, Semantic) */}
        <section
          aria-labelledby="search-heading"
          className="bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-800 rounded-lg p-6 shadow-sm space-y-4"
        >
          <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
            <div>
              <h2 id="search-heading" className="text-base font-semibold text-gray-900 dark:text-gray-100 flex items-center gap-2">
                <span>Policy Retrieval</span>
                <span className="text-[11px] font-normal px-2 py-0.5 rounded bg-purple-100 text-purple-800 dark:bg-purple-900/40 dark:text-purple-300">
                  Cross-Encoder
                </span>
                <span className="text-[11px] font-normal px-2 py-0.5 rounded bg-emerald-100 text-emerald-800 dark:bg-emerald-900/40 dark:text-emerald-300">
                  BM25 + RRF
                </span>
                <span className="text-[11px] font-normal px-2 py-0.5 rounded bg-blue-100 text-blue-800 dark:bg-blue-900/40 dark:text-blue-300">
                  Qdrant Vector
                </span>
              </h2>
              <p className="text-xs text-gray-500 dark:text-gray-400 mt-0.5">
                Query internal policies with semantic vector search, BM25 keyword matching, or cross-encoder reranked hybrid search.
              </p>
            </div>

            {/* Search Mode Toggle */}
            <div className="flex items-center gap-2">
              <span className="text-xs text-gray-500 dark:text-gray-400 font-medium">Mode:</span>
              <div className="inline-flex p-0.5 bg-gray-100 dark:bg-gray-800 rounded-md text-xs border border-gray-200 dark:border-gray-700">
                <button
                  id="search-mode-rerank-button"
                  type="button"
                  onClick={() => handleModeChange("rerank")}
                  className={`px-3 py-1 rounded transition-colors ${
                    searchMode === "rerank"
                      ? "bg-white dark:bg-gray-900 text-purple-600 dark:text-purple-400 font-medium shadow-xs"
                      : "text-gray-500 hover:text-gray-900 dark:hover:text-gray-200"
                  }`}
                >
                  Reranked
                </button>
                <button
                  id="search-mode-hybrid-button"
                  type="button"
                  onClick={() => handleModeChange("hybrid")}
                  className={`px-3 py-1 rounded transition-colors ${
                    searchMode === "hybrid"
                      ? "bg-white dark:bg-gray-900 text-blue-600 dark:text-blue-400 font-medium shadow-xs"
                      : "text-gray-500 hover:text-gray-900 dark:hover:text-gray-200"
                  }`}
                >
                  Hybrid (RRF)
                </button>
                <button
                  id="search-mode-semantic-button"
                  type="button"
                  onClick={() => handleModeChange("semantic")}
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
            {searchMode === "rerank"
              ? "Reranked mode retrieves candidates using BM25 and vector hybrid search (RRF), then rescores top passages using a cross-encoder model for deep semantic relevance."
              : searchMode === "hybrid"
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
                  searchMode === "rerank"
                    ? "Ask in natural language or enter exact policy codes (e.g., annual leave allowance, travel expenses, HR-2024)..."
                    : searchMode === "hybrid"
                      ? "Ask in natural language or enter exact policy codes (e.g., HR-2024, annual leave, business travel)..."
                      : "Ask or search policies (e.g., annual leave allowance, business travel expenses, health insurance)..."
                }
                value={searchQuery}
                onChange={(e) => {
                  setSearchQuery(e.target.value);
                  if (searchResults !== null) {
                    setSearchResults(null);
                  }
                }}
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
                  ) : searchMode === "rerank" ? (
                    "Rerank Search"
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
                          {executedMode === "rerank" ? (
                            <>
                              {result.rerank_score !== undefined && (
                                <span
                                  className="inline-flex items-center px-1.5 py-0.5 rounded font-mono font-medium bg-purple-100 text-purple-800 dark:bg-purple-900/40 dark:text-purple-300"
                                  title="Cross-encoder relevance score"
                                >
                                  Rerank: {result.rerank_score.toFixed(6)}
                                </span>
                              )}
                              {result.rrf_score !== undefined && (
                                <span
                                  className="inline-flex items-center px-1.5 py-0.5 rounded font-mono text-[10px] bg-emerald-100 text-emerald-800 dark:bg-emerald-900/40 dark:text-emerald-300"
                                  title="Retrieval-stage Reciprocal Rank Fusion score"
                                >
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
                          ) : executedMode === "hybrid" ? (
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
                          <span className="text-gray-300 dark:text-gray-700">|</span>
                          <button
                            onClick={() => handleDeleteDocument(doc.document_id, doc.filename)}
                            disabled={deletingDocId === doc.document_id}
                            className="text-xs text-red-600 hover:text-red-800 dark:text-red-400 dark:hover:text-red-300 font-medium disabled:opacity-50"
                            title="Permanently delete document"
                          >
                            {deletingDocId === doc.document_id ? "Deleting…" : "Delete"}
                          </button>
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
                {docDetail && (
                  <button
                    onClick={() => handleDeleteDocument(docDetail.document_id, docDetail.filename)}
                    disabled={deletingDocId === docDetail.document_id}
                    className="text-xs px-3 py-1 bg-red-50 dark:bg-red-950/40 text-red-600 dark:text-red-400 hover:bg-red-100 dark:hover:bg-red-900/50 rounded font-medium disabled:opacity-50 transition-colors"
                    title="Permanently delete document"
                  >
                    {deletingDocId === docDetail.document_id ? "Deleting…" : "Delete Policy"}
                  </button>
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
                              (No extractable text found on this page. If this page is an image or scanned document, OCR processing is required.)
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
