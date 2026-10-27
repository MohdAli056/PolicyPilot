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
        `Document "${data.filename}" successfully ingested: ${data.total_pages} page(s), ${data.total_characters} characters extracted.`
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
  };

  const closeDetail = () => {
    setSelectedDocId(null);
    setDocDetail(null);
    setDetailError(null);
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
                AI-powered company policy assistant — Phase 2: Ingestion & Text Extraction
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

        {/* Upload Section */}
        <section aria-labelledby="upload-heading" className="bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-800 rounded-lg p-6 shadow-sm">
          <h2 id="upload-heading" className="text-base font-semibold text-gray-900 dark:text-gray-100 mb-2">
            Upload Policy PDF
          </h2>
          <p className="text-xs text-gray-500 dark:text-gray-400 mb-4">
            Upload a policy document in PDF format. Text will be extracted page-by-page preserving 1-based page numbers.
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
                {uploading ? "Extracting Pages…" : "Upload & Ingest"}
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
                    <th className="py-2.5 px-3">Uploaded</th>
                    <th className="py-2.5 px-3 text-right">Actions</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-100 dark:divide-gray-800/60">
                  {documents.map((doc) => (
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
                      <td className="py-2.5 px-3 text-gray-500 dark:text-gray-400 whitespace-nowrap">
                        {new Date(doc.uploaded_at).toLocaleString()}
                      </td>
                      <td className="py-2.5 px-3 text-right space-x-2 whitespace-nowrap">
                        <button
                          onClick={() => openDetail(doc.document_id)}
                          className="text-xs text-blue-600 hover:text-blue-800 dark:text-blue-400 dark:hover:text-blue-300 font-medium"
                        >
                          View Extracted Text
                        </button>
                        <span className="text-gray-300 dark:text-gray-700">|</span>
                        <a
                          href={`${API_URL}/api/documents/${doc.document_id}/file`}
                          target="_blank"
                          rel="noreferrer"
                          className="text-xs text-gray-600 hover:text-gray-900 dark:text-gray-400 dark:hover:text-gray-200"
                        >
                          Original PDF
                        </a>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </section>

        {/* Document Detail View */}
        {selectedDocId && (
          <section
            aria-labelledby="detail-heading"
            id="document-detail-panel"
            className="bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-800 rounded-lg p-6 shadow-sm space-y-4"
          >
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
              <div className="flex items-center gap-3">
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
                Fetching extracted page text…
              </p>
            ) : detailError ? (
              <div className="text-xs text-red-600 dark:text-red-400 p-3 bg-red-50 dark:bg-red-950/40 rounded">
                {detailError}
              </div>
            ) : docDetail ? (
              <div className="space-y-4">
                <div className="flex items-center gap-4 text-xs text-gray-600 dark:text-gray-400 bg-gray-50 dark:bg-gray-800/40 p-3 rounded">
                  <span>
                    Total Pages: <strong className="text-gray-900 dark:text-gray-100">{docDetail.total_pages}</strong>
                  </span>
                  <span>
                    Total Characters: <strong className="text-gray-900 dark:text-gray-100">{docDetail.total_characters.toLocaleString()}</strong>
                  </span>
                  <span>
                    Status: <strong className="text-gray-900 dark:text-gray-100">{docDetail.ingestion_status}</strong>
                  </span>
                </div>

                {/* Page by page extracted text */}
                <div className="space-y-4">
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
                            (No extractable text found on this page. If this page is an image or scanned document, OCR is not supported in Phase 2.)
                          </p>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            ) : null}
          </section>
        )}
      </div>
    </main>
  );
}
