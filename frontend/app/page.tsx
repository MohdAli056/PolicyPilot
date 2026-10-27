"use client";

import { useEffect, useState } from "react";

/** Backend API base URL, configurable via environment variable. */
const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

type HealthStatus = "loading" | "healthy" | "unavailable";

interface HealthData {
  status: string;
  service: string;
  version: string;
  timestamp: string;
}

export default function Home() {
  const [healthStatus, setHealthStatus] = useState<HealthStatus>("loading");
  const [healthData, setHealthData] = useState<HealthData | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
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
        setError(null);
      })
      .catch((err) => {
        setHealthStatus("unavailable");
        setError(
          err instanceof DOMException && err.name === "AbortError"
            ? "Request timed out"
            : err instanceof Error
              ? err.message
              : "Unknown error"
        );
      })
      .finally(() => clearTimeout(timeout));

    return () => {
      controller.abort();
      clearTimeout(timeout);
    };
  }, []);

  return (
    <main className="min-h-screen flex flex-col items-center justify-center px-4">
      <div className="w-full max-w-lg text-center">
        {/* Header */}
        <h1 className="text-3xl font-bold tracking-tight mb-2">PolicyPilot</h1>
        <p className="text-sm text-gray-500 dark:text-gray-400 mb-8">
          AI-powered company policy assistant. Ask questions about leave,
          insurance, travel, and reimbursement policies — and get
          evidence-grounded answers with citations.
        </p>

        {/* Health status card */}
        <div className="border rounded-lg p-6 text-left bg-white dark:bg-gray-900 dark:border-gray-700 shadow-sm">
          <h2 className="text-sm font-semibold uppercase tracking-wide text-gray-400 dark:text-gray-500 mb-4">
            Backend Status
          </h2>

          {healthStatus === "loading" && (
            <div className="flex items-center gap-3">
              <span className="relative flex h-3 w-3">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-yellow-400 opacity-75"></span>
                <span className="relative inline-flex rounded-full h-3 w-3 bg-yellow-500"></span>
              </span>
              <span className="text-sm text-gray-600 dark:text-gray-300">
                Connecting to backend…
              </span>
            </div>
          )}

          {healthStatus === "healthy" && healthData && (
            <div className="space-y-3">
              <div className="flex items-center gap-3">
                <span className="inline-flex rounded-full h-3 w-3 bg-green-500"></span>
                <span className="text-sm font-medium text-green-700 dark:text-green-400">
                  Healthy
                </span>
              </div>
              <dl className="grid grid-cols-2 gap-x-4 gap-y-2 text-sm">
                <dt className="text-gray-500 dark:text-gray-400">Service</dt>
                <dd className="text-gray-900 dark:text-gray-100">
                  {healthData.service}
                </dd>
                <dt className="text-gray-500 dark:text-gray-400">Version</dt>
                <dd className="text-gray-900 dark:text-gray-100">
                  {healthData.version}
                </dd>
                <dt className="text-gray-500 dark:text-gray-400">Timestamp</dt>
                <dd className="text-gray-900 dark:text-gray-100 break-all">
                  {healthData.timestamp}
                </dd>
              </dl>
            </div>
          )}

          {healthStatus === "unavailable" && (
            <div className="space-y-2">
              <div className="flex items-center gap-3">
                <span className="inline-flex rounded-full h-3 w-3 bg-red-500"></span>
                <span className="text-sm font-medium text-red-700 dark:text-red-400">
                  Unavailable
                </span>
              </div>
              <p className="text-xs text-gray-500 dark:text-gray-400">
                Could not reach the backend at{" "}
                <code className="bg-gray-100 dark:bg-gray-800 px-1 py-0.5 rounded text-xs">
                  {API_URL}
                </code>
              </p>
              {error && (
                <p className="text-xs text-red-600 dark:text-red-400">
                  {error}
                </p>
              )}
            </div>
          )}
        </div>

        <p className="mt-6 text-xs text-gray-400 dark:text-gray-600">
          Phase 1 — Foundation verified
        </p>
      </div>
    </main>
  );
}
