/**
 * TrashTheTest API Services
 * 
 * This module provides frontend-friendly API functions that:
 * - Call backend endpoints
 * - Transform responses to match frontend component expectations
 * - Handle loading and error states via React Query
 * 
 * IMPORTANT: This adapts data in the API layer, not the UI components.
 */

import { setBaseUrl, setAuthTokenGetter } from "@workspace/api-client-react";

// ============================================================
// Configuration
// ============================================================

// Get backend URL from environment variables. The fallback keeps direct
// Node-based smoke tests possible; Vite still injects import.meta.env at build.
const VITE_ENV = import.meta.env ?? {};
const API_BASE_URL = VITE_ENV.VITE_API_BASE_URL || "http://localhost:8000";

// Configure the base URL for all API calls
setBaseUrl(API_BASE_URL);

// Auth storage is intentionally centralized. VITE_AUTH_TOKEN_KEY is used for
// the access token; the refresh token has a stable server-compat key.
const ACCESS_TOKEN_KEY =
  VITE_ENV.VITE_AUTH_TOKEN_KEY || "trash_the_test_token";
const REFRESH_TOKEN_KEY = "trash_the_test_refresh_token";
const USER_CACHE_KEY = "user";

// Auth token getter - retrieves token from localStorage
const getAuthToken = () => {
  if (typeof window === "undefined") return null;
  return localStorage.getItem(ACCESS_TOKEN_KEY);
};

const clearStoredAuthSession = () => {
  if (typeof window === "undefined") return;
  localStorage.removeItem(ACCESS_TOKEN_KEY);
  localStorage.removeItem(REFRESH_TOKEN_KEY);
  localStorage.removeItem(USER_CACHE_KEY);
};

// Configure auth token getter
setAuthTokenGetter(getAuthToken);

// Export the base URL for use in other parts of the app
export const API_BASE = API_BASE_URL;

// ============================================================
// Internal Fetch Helper
// ============================================================

export interface FetchOptions extends RequestInit {
  responseType?: "json" | "text" | "blob";
  /**
   * Client-side deadline for the whole call (headers + body). When it fires
   * the request is aborted and apiFetch rejects with `kind: "timeout"`.
   */
  timeoutMs?: number;
}

type ApiHttpError = Error & {
  status?: number;
  authRefreshFailed?: boolean;
};

const authEndpointWithoutRefresh = new Set([
  "/api/auth/login",
  "/api/auth/register",
  "/api/auth/refresh",
  "/api/auth/logout",
]);

const isCredentialEndpoint = (path: string): boolean =>
  authEndpointWithoutRefresh.has(path.split("?", 1)[0]);

async function httpErrorFromResponse(
  response: Response,
  authRefreshFailed = false
): Promise<ApiHttpError> {
  let errorData: unknown;
  try {
    errorData = await response.json();
  } catch {
    errorData = { detail: response.statusText };
  }
  const detail =
    typeof errorData === "object" && errorData !== null
      ? (errorData as { detail?: unknown }).detail
      : undefined;
  const httpError = new Error(
    typeof detail === "string" && detail ? detail : `HTTP ${response.status}`
  ) as ApiHttpError;
  httpError.status = response.status;
  httpError.authRefreshFailed = authRefreshFailed;
  return httpError;
}

async function apiFetch<T = unknown>(
  path: string,
  options: FetchOptions = {}
): Promise<T> {
  const { responseType = "json", timeoutMs, ...fetchOptions } = options;
  const skipAuthRefresh = isCredentialEndpoint(path);
  const token = skipAuthRefresh ? null : getAuthToken();
  const headers = new Headers(fetchOptions.headers);

  if (token) {
    headers.set("Authorization", `Bearer ${token}`);
  }

  // Set default content-type for JSON requests
  if (fetchOptions.body && typeof fetchOptions.body === "string" && !headers.has("content-type")) {
    headers.set("content-type", "application/json");
  }

  // Optional per-call deadline (ADR-008): the controller is shared by the
  // original attempt and the single 401-refresh retry below.
  let controller: AbortController | null = null;
  let timeoutId: ReturnType<typeof setTimeout> | null = null;
  if (timeoutMs && timeoutMs > 0) {
    controller = new AbortController();
    timeoutId = setTimeout(
      () => controller?.abort(new DOMException("timeout", "AbortError")),
      timeoutMs
    );
  }
  const clearTimer = () => {
    if (timeoutId !== null) {
      clearTimeout(timeoutId);
      timeoutId = null;
    }
  };

  const request = (): Promise<Response> =>
    fetch(`${API_BASE_URL}${path}`, {
      ...fetchOptions,
      // The backend explicitly allows credentialed CORS. Bearer auth is still
      // required; include keeps the transport compatible with cookie-backed
      // deployments without weakening the explicit origin allowlist.
      credentials: fetchOptions.credentials ?? "include",
      headers,
      signal: controller?.signal ?? fetchOptions.signal ?? null,
    });

  try {
    let response = await request();
    let refreshFailed = false;

    // One 401 may mean only that the short-lived access token expired while
    // several reads were already in flight. Refresh once (shared single-flight),
    // replace the Authorization header, and retry the original request exactly
    // once. A second 401 is surfaced without clearing the renewed session.
    if (response.status === 401 && token && !skipAuthRefresh) {
      const renewed = await refreshAccessToken(token);
      if (renewed) {
        headers.set("Authorization", `Bearer ${renewed}`);
        response = await request();
      } else {
        refreshFailed = true;
      }
    }

    if (!response.ok) {
      throw await httpErrorFromResponse(response, refreshFailed);
    }

    if (responseType === "text") {
      return (await response.text()) as T;
    }

    if (responseType === "blob") {
      return (await response.blob()) as T;
    }

    return (await response.json()) as T;
  } catch (err) {
    const aborted =
      controller?.signal.aborted === true ||
      (((err as Error)?.name ?? "") === "AbortError" &&
        typeof timeoutMs === "number" &&
        timeoutMs > 0);
    if (aborted) {
      const seconds = Math.round((timeoutMs ?? 0) / 1000);
      throw Object.assign(
        new Error(
          `Request to ${path} timed out after ${seconds}s (${timeoutMs}ms)`
        ),
        { status: 0, kind: "timeout" }
      ) as ApiHttpError;
    }
    throw err;
  } finally {
    // Keep the timer armed until the body is read so a stalled response.json()
    // still trips the deadline; clear it exactly once on the way out.
    clearTimer();
  }
}

// ============================================================
// Types - Frontend-facing interfaces
// ============================================================

// Project types
export interface Project {
  id: string;
  name: string;
  description?: string;
  status: "processing" | "ready" | "failed";
  progress: number;
  createdAt: string;
  updatedAt: string;
  documentCount?: number;
  lastActivity?: string;
}

// Document types
export interface Document {
  id: string;
  projectId: string;
  fileName: string;
  fileType: string;
  sourceType?: "study_material" | "syllabus" | "question_paper" | "question_bank" | "reference";
  filePath: string;
  pageCount?: number;
  processingStatus: "pending" | "processing" | "completed" | "failed";
  processingError?: string;
  isOcrRequired: boolean;
  createdAt: string;
  updatedAt: string;
  // Provenance (backend T110/T210 fields, camelCase aliases).
  mimeType?: string;
  fileSize?: number;
  checksum?: string;
  indexingState?: string;
  pageMethods?: Record<string, { method?: string; count?: number } | string> | null;
  pageCountConfirmed?: number;
}

export type DocumentSourceType =
  | "study_material"
  | "syllabus"
  | "question_paper"
  | "question_bank"
  | "reference";

export interface DocumentUploadResponse {
  documentId: string;
  fileName: string;
  fileType: string;
  sourceType?: DocumentSourceType;
  status: string;
  message: string;
  jobId?: string;
}

export interface DocumentProgress {
  documentId: string;
  status: string;
  progress: number;
  stage?: string;
  message?: string;
}

// Job types (normalised from the backend's snake_case wire contract)
export interface Job {
  id: string;
  jobType: string;
  status: "pending" | "processing" | "completed" | "failed" | "cancelled";
  progress: number;
  stage?: string;
  stageMessage?: string;
  errorMessage?: string;
  output_data?: Record<string, unknown> | null;
  createdAt: string;
  startedAt?: string;
  completedAt?: string;
}

type RawJob = Record<string, unknown>;

const optionalString = (value: unknown): string | undefined =>
  value === undefined || value === null || value === "" ? undefined : String(value);

const toJob = (raw: RawJob): Job => {
  const status = String(raw.status ?? "pending") as Job["status"];
  return {
    id: String(raw.id ?? raw.jobId ?? raw.job_id ?? ""),
    jobType: String(raw.jobType ?? raw.job_type ?? ""),
    status,
    progress: Number(raw.progress ?? 0) || 0,
    stage: optionalString(raw.stage),
    stageMessage: optionalString(raw.stageMessage ?? raw.stage_message),
    errorMessage: optionalString(raw.errorMessage ?? raw.error_message),
    output_data:
      typeof raw.output_data === "object" && raw.output_data !== null
        ? (raw.output_data as Record<string, unknown>)
        : null,
    createdAt: String(raw.createdAt ?? raw.created_at ?? ""),
    startedAt: optionalString(raw.startedAt ?? raw.started_at),
    completedAt: optionalString(raw.completedAt ?? raw.completed_at),
  };
};

// Study Session types
export interface StudySession {
  sessionId: string;
  sessionType: "flashcards" | "quiz" | "notes" | "tutor";
  status: string;
  totalItems: number;
  currentItem?: StudyItem;
}

export interface StudyItem {
  id: string;
  type: "flashcard" | "quiz" | "note";
  front: string;
  back?: string;
  question?: string;
  options?: string[];
  correctAnswer?: string;
}

export interface StudyResponse {
  itemId: string;
  isCorrect: boolean;
  correctAnswer?: string;
  feedback: string;
  nextItem?: StudyItem;
}

export interface SessionResult {
  sessionId: string;
  sessionType: string;
  totalItems: number;
  correctCount: number;
  incorrectCount: number;
  skippedCount: number;
  score: number;
  totalDurationMs: number;
  avgTimePerItem: number;
}

// Content generation types
export interface GeneratedContent {
  projectId: string;
  mode: string;
  content: unknown;
  cached: boolean;
  processingTime: number;
}

export interface FlashcardDeck {
  id: string;
  projectId: string;
  name: string;
  cards: Flashcard[];
  cardCount: number;
}

export interface Flashcard {
  id: string;
  front: string;
  back: string;
  difficulty?: "easy" | "medium" | "hard";
}

export interface Quiz {
  id: string;
  projectId: string;
  title: string;
  questions: QuizQuestion[];
}

export interface QuizQuestion {
  id: string;
  question: string;
  options: string[];
  correctAnswer: number;
  explanation?: string;
}

export interface Note {
  id: string;
  projectId: string;
  title: string;
  content: string;
  createdAt: string;
}

// AI Tutor types
export interface Citation {
  chunk_id: string;
  document_id?: string;
  page?: number | null;
  span?: string;
  method?: string;
}

export const citationLabel = (c: Citation, index: number): string => {
  // Render chunk citation span/page/method, never a bare id.
  const parts: string[] = [`chunk ${c.chunk_id.slice(0, 8)}`];
  if (c.page !== undefined && c.page !== null) parts.push(`p.${c.page}`);
  if (c.method) parts.push(c.method);
  if (c.span) parts.push(`"${c.span.slice(0, 60)}${c.span.length > 60 ? "..." : ""}"`);
  return parts.length > 1 ? parts.join(" · ") : `Source ${index + 1}`;
};

export interface TutorMessage {
  role: "user" | "assistant";
  content: string;
  sources?: string[];
  citations?: Citation[];
  context?: string;
}

export interface TutorResponse {
  response: string;
  sources?: string[];
  citations?: Citation[];
  context?: string;
  intent?: "conversational" | "academic" | "project_specific" | "unsupported" | "ambiguous";
  retrieval_used?: boolean;
  grounded?: boolean;
}

// User/Auth types
export interface User {
  id: string;
  email: string;
  name: string;
  settings?: Record<string, unknown>;
}

export interface AuthResponse {
  user: User;
  token: string;
}

// ============================================================
// API Functions - Backend communication
// ============================================================

// Health check
export const healthCheck = async (): Promise<{ status: string; service: string }> => {
  return apiFetch<{ status: string; service: string }>("/health");
};

// ============================================================
// Project API
// ============================================================

interface CreateProjectRequest {
  name: string;
  description?: string;
}

// Backend contract: POST/GET /api/projects speak {title, created_at,
// updated_at}; the UI speaks {name, createdAt, updatedAt}. Normalise in
// this API layer (never in components) and preserve the backend schema.
type RawProject = Record<string, unknown>;

const toProject = (raw: RawProject): Project => {
  const pick = (...keys: string[]): unknown => {
    for (const key of keys) {
      const value = raw[key];
      if (value !== undefined && value !== null) return value;
    }
    return undefined;
  };
  const status = pick("status");
  const progress = pick("progress");
  return {
    id: String(pick("id") ?? ""),
    name: String(pick("name", "title") ?? ""),
    description: (pick("description") as string | undefined) ?? undefined,
    status: (status === "processing" || status === "ready" || status === "failed"
      ? status
      : "processing") as Project["status"],
    progress: typeof progress === "number" ? progress : Number(progress ?? 0) || 0,
    createdAt: String(pick("createdAt", "created_at") ?? ""),
    updatedAt: String(pick("updatedAt", "updated_at") ?? ""),
    documentCount: pick("documentCount", "document_count") as number | undefined,
    lastActivity: pick("lastActivity", "last_activity") as string | undefined,
  };
};

export const createProject = async (request: CreateProjectRequest): Promise<Project> => {
  // Call the real API to create a project
  try {
    const name = request.name.trim();
    const response = await apiFetch<RawProject>("/api/projects", {
      method: "POST",
      // Backend accepts title or name; send both so either contract works.
      body: JSON.stringify({ title: name, name, description: request.description }),
    });
    const project = toProject(response);
    console.log("[API] Project created:", project);
    return project;
  } catch (error) {
    console.error("[API Error] Failed to create project:", error);
    throw error;
  }
};

export const getProjects = async (): Promise<Project[]> => {
  // Call the real API to get projects from the backend
  try {
    const response = await apiFetch<{ projects: RawProject[]; total: number }>("/api/projects");
    console.log("[API] Projects fetched:", response);
    return (response.projects || []).map(toProject);
  } catch (error) {
    console.error("[API Error] Failed to fetch projects:", error);
    // Return empty array instead of mock data - let UI show empty state
    return [];
  }
};

// Default project naming (Study first-run onboarding).
//
// On first load with zero projects the app auto-creates exactly one
// project via the real POST /api/projects. The name derives from
// GET /api/auth/me: "<firstNameOrUsername>'s project" (e.g.
// "test's project", "john's project"); "My First Project" when the
// identity carries no usable name or email prefix.
export const defaultProjectNameFor = (
  user: { name?: string | null; email?: string | null } | null | undefined
): string => {
  const displayName = (user?.name ?? "").trim();
  let first = displayName.split(/\s+/).filter(Boolean)[0] ?? "";
  if (!first) {
    const prefix = (user?.email ?? "").split("@")[0] ?? "";
    first = prefix.trim();
  }
  // Keep the possessive readable: first alphanumeric run, lowercase
  // ("John", "john.doe", "john_doe42" -> "john", "john", "john").
  first = (first.match(/[a-z0-9]+/i)?.[0] ?? "").toLowerCase();
  if (!first) return "My First Project";
  return `${first}'s project`;
};

export const getProject = async (projectId: string): Promise<Project | null> => {
  // Call the real API to get a specific project
  try {
    const response = await apiFetch<RawProject | { project: RawProject }>(`/api/projects/${projectId}`);
    const raw = (response as { project?: RawProject }).project ?? (response as RawProject);
    if (!raw || typeof raw !== "object" || !(raw as RawProject).id) return null;
    const project = toProject(raw as RawProject);
    console.log("[API] Project fetched:", project);
    return project;
  } catch (error) {
    console.error("[API Error] Failed to fetch project:", error);
    return null;
  }
};

export const deleteProject = async (projectId: string): Promise<boolean> => {
  // Call the real API to delete a project
  try {
    await apiFetch(`/api/projects/${projectId}`, { method: "DELETE" });
    console.log("[API] Project deleted:", projectId);
    return true;
  } catch (error) {
    console.error("[API Error] Failed to delete project:", error);
    throw error;
  }
};

// ============================================================
// Document API
// ============================================================

export const uploadDocument = async (
  file: File,
  projectId: string,
  opts?: { idempotencyKey?: string; signal?: AbortSignal; sourceType?: DocumentSourceType }
): Promise<DocumentUploadResponse> => {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("project_id", projectId);
  formData.append("source_type", opts?.sourceType ?? "study_material");

  const token = getAuthToken();
  const headers: Record<string, string> = {};
  if (token) {
    headers["Authorization"] = `Bearer ${token}`;
  }
  if (opts?.idempotencyKey) {
    headers["Idempotency-Key"] = opts.idempotencyKey;
  }

  const response = await fetch(`${API_BASE_URL}/api/documents/upload`, {
    method: "POST",
    headers,
    body: formData,
    credentials: "include",
    signal: opts?.signal,
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: "Upload failed" }));
    throw new Error(error.detail || "Upload failed");
  }

  return response.json();
};

export const streamUploadDocument = async (
  file: File,
  projectId: string,
  opts?: { idempotencyKey?: string; signal?: AbortSignal; sourceType?: DocumentSourceType }
): Promise<DocumentUploadResponse> => {
  // Large-file path: raw octet-stream with project/file headers so the
  // server can stream to storage without buffering the whole body.
  const token = getAuthToken();
  const headers: Record<string, string> = {
    "Content-Type": "application/octet-stream",
    "X-Project-Id": projectId,
    "X-File-Name": file.name,
    "X-Source-Type": opts?.sourceType ?? "study_material",
  };
  if (token) {
    headers["Authorization"] = `Bearer ${token}`;
  }
  if (opts?.idempotencyKey) {
    headers["Idempotency-Key"] = opts.idempotencyKey;
  }

  const response = await fetch(`${API_BASE_URL}/api/documents/upload/stream`, {
    method: "POST",
    headers,
    body: file,
    credentials: "include",
    signal: opts?.signal,
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: "Upload failed" }));
    throw new Error(error.detail || "Upload failed");
  }

  return response.json();
};

export const getDocument = async (documentId: string): Promise<Document> => {
  return apiFetch<Document>(`/api/documents/${documentId}`);
};

export const getDocumentProgress = async (documentId: string): Promise<DocumentProgress> => {
  return apiFetch<DocumentProgress>(`/api/documents/${documentId}/progress`);
};

export const getProjectDocuments = async (projectId: string): Promise<Document[]> => {
  const response = await apiFetch<{ documents: Document[]; total: number }>(
    `/api/documents/project/${projectId}`
  );
  return response.documents;
};

export const deleteDocument = async (documentId: string): Promise<boolean> => {
  await apiFetch(`/api/documents/${documentId}`, { method: "DELETE" });
  return true;
};

export const fetchDocumentBytes = async (documentId: string): Promise<Blob> => {
  // Raw source bytes for the PDF viewer (GET /api/documents/{id}/file,
  // JWT ownership enforced server-side). Fetched as a Blob so the viewer
  // can render without ever exposing the token in a URL.
  const token = getAuthToken();
  const response = await fetch(
    `${API_BASE_URL}/api/documents/${encodeURIComponent(documentId)}/file`,
    { headers: token ? { Authorization: `Bearer ${token}` } : {} }
  );
  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: "Document unavailable" }));
    throw new Error(error.detail || "Document unavailable");
  }
  return response.blob();
};

export interface MultiUploadItem {
  fileName: string | null;
  documentId: string | null;
  jobId: string | null;
  status: string;
  error: string | null;
}

export const uploadDocumentsMulti = async (
  files: File[],
  projectId: string,
  sourceType: DocumentSourceType = "study_material"
): Promise<MultiUploadItem[]> => {
  // Multi-file wrapper -> POST /api/documents/upload/multi (per-file jobs).
  const formData = new FormData();
  for (const file of files) {
    formData.append("files", file);
  }
  formData.append("project_id", projectId);
  formData.append("source_type", sourceType);

  const token = getAuthToken();
  const headers: Record<string, string> = {};
  if (token) {
    headers["Authorization"] = `Bearer ${token}`;
  }

  const response = await fetch(`${API_BASE_URL}/api/documents/upload/multi`, {
    method: "POST",
    headers,
    body: formData,
    credentials: "include",
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: "Multi upload failed" }));
    throw new Error(error.detail || "Multi upload failed");
  }

  return response.json();
};

// ============================================================
// Job API
// ============================================================

export const getJobStatus = async (jobId: string): Promise<Job> => {
  const raw = await apiFetch<RawJob>(`/api/jobs/${jobId}`);
  return toJob(raw);
};

export const getJobProgress = async (jobId: string): Promise<Job> => {
  const raw = await apiFetch<RawJob>(`/api/jobs/${jobId}/progress`);
  return toJob(raw);
};

export const cancelJob = async (jobId: string): Promise<boolean> => {
  await apiFetch(`/api/jobs/${jobId}/cancel`, { method: "POST" });
  return true;
};

export const streamJobProgress = (
  jobId: string,
  onEvent: (job: Job) => void,
  onError?: (err: Error) => void
): (() => void) => {
  // Real SSE stream from GET /api/jobs/{id}/stream via fetch (EventSource
  // cannot send the Authorization header, so we parse the SSE byte stream
  // ourselves). A long upload can outlive the access-token TTL: on a 401 we
  // refresh once (shared single-flight helper) and reconnect instead of
  // going silent; if the refresh fails we surface a real auth error so the
  // caller can send the user back to sign-in (rather than an empty spinner).
  const controller = new AbortController();
  (async () => {
    let refreshed = false;
    try {
      for (;;) {
        const token = getAuthToken();
        const response = await fetch(`${API_BASE_URL}/api/jobs/${jobId}/stream`, {
          headers: token ? { Authorization: `Bearer ${token}` } : {},
          credentials: "include",
          signal: controller.signal,
        });

        if (response.status === 401) {
          if (refreshed) {
            const authError = Object.assign(
              new Error("Session expired while streaming job progress."),
              { status: 401, authRefreshFailed: false }
            );
            onError?.(authError);
            return;
          }
          refreshed = true;
          const renewed = await refreshAccessToken(token ?? undefined);
          if (!renewed) {
            const authError = Object.assign(
              new Error(
                "Session expired while streaming job progress. Sign in again to continue."
              ),
              { status: 401, authRefreshFailed: true }
            );
            onError?.(authError);
            return;
          }
          continue; // reconnect once with the renewed token
        }

        if (!response.ok || !response.body) {
          throw new Error(`Job stream failed: HTTP ${response.status}`);
        }
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";
        for (;;) {
          const { done, value } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });
          const parts = buffer.split("\n\n");
          buffer = parts.pop() ?? "";
          for (const part of parts) {
            const line = part.split("\n").find((l) => l.startsWith("data:"));
            if (!line) continue;
            try {
              onEvent(JSON.parse(line.slice(5).trim()) as Job);
            } catch (err) {
              onError?.(err instanceof Error ? err : new Error("Bad SSE payload"));
            }
          }
        }
        return;
      }
    } catch (err) {
      if ((err as Error)?.name !== "AbortError") {
        onError?.(err instanceof Error ? err : new Error("Job stream failed"));
      }
    }
  })();
  return () => controller.abort();
};

export const DEFAULT_JOB_POLL_TIMEOUT_MS = 30 * 60 * 1000;

// ADR-008 (T013/T014): per-call client deadlines so one stalled LLM call
// cannot hang the Website builder indefinitely. The Website page overrides
// explanation down to 60s because it is best-effort.
export const DEFAULT_GENERATION_TIMEOUT_MS = 180_000;
export const DEFAULT_WEBSITE_TIMEOUT_MS = 240_000;
// Practice includes a cold model load on the first call after an API restart
// (Qwen embedding + bge-reranker). The backend bounds itself at 180s and
// returns a typed 504; the client waits longer so that honest error, not a
// premature abort, is what the user sees.
export const PRACTICE_GENERATION_TIMEOUT_MS = 300_000;

export class JobRunError extends Error {
  readonly jobId: string;
  readonly lastJob?: Job;

  constructor(message: string, jobId: string, lastJob?: Job) {
    super(message);
    this.name = "JobRunError";
    this.jobId = jobId;
    this.lastJob = lastJob;
  }
}

export const pollJobUntilDone = async (
  jobId: string,
  opts?: {
    intervalMs?: number;
    timeoutMs?: number;
    signal?: AbortSignal;
    onUpdate?: (job: Job) => void;
  }
): Promise<Job> => {
  // Poll the real job API until a terminal state. The progress endpoint is
  // intentionally lightweight; terminal failures are hydrated once from the
  // full job row so the UI receives its real error and output payload.
  const intervalMs = opts?.intervalMs ?? 1500;
  const timeoutMs = opts?.timeoutMs ?? DEFAULT_JOB_POLL_TIMEOUT_MS;
  const started = Date.now();
  let lastJob: Job | undefined;

  for (;;) {
    if (opts?.signal?.aborted) {
      throw new JobRunError("Job polling was cancelled.", jobId, lastJob);
    }
    const progressJob = await getJobProgress(jobId);
    lastJob = progressJob;
    opts?.onUpdate?.(progressJob);

    if (
      progressJob.status === "completed" ||
      progressJob.status === "failed" ||
      progressJob.status === "cancelled"
    ) {
      if (progressJob.status === "completed") return progressJob;
      const fullJob = await getJobStatus(jobId).catch(() => progressJob);
      opts?.onUpdate?.(fullJob);
      return fullJob;
    }

    if (Date.now() - started >= timeoutMs) {
      const stage = lastJob?.stage || lastJob?.status || "unknown stage";
      throw new JobRunError(
        `Timed out after ${Math.round(timeoutMs / 60000)} minutes waiting for job ${jobId} (${stage}). The job is still recorded as ${lastJob?.status ?? "unknown"}; retry this job to continue waiting.`,
        jobId,
        lastJob
      );
    }

    await new Promise((resolve) => setTimeout(resolve, intervalMs));
  }
};

// ============================================================
// Generation API
// ============================================================

interface GenerateNotesRequest {
  projectId: string;
  query?: string;
  enableResearch?: boolean;
  timeoutMs?: number;
}

interface GenerateFlashcardsRequest {
  projectId: string;
  query?: string;
  enableResearch?: boolean;
  timeoutMs?: number;
}

interface GenerateQuizRequest {
  projectId: string;
  query?: string;
  enableResearch?: boolean;
  timeoutMs?: number;
}

interface GenerateSummaryRequest {
  projectId: string;
  query?: string;
  enableResearch?: boolean;
}

interface GenerateWebsiteRequest {
  projectId: string;
  query?: string;
  enableResearch?: boolean;
}

export const generateNotes = async (request: GenerateNotesRequest): Promise<GeneratedContent> => {
  return apiFetch<GeneratedContent>("/api/generation/notes", {
    method: "POST",
    body: JSON.stringify({
      project_id: request.projectId,
      query: request.query ?? null,
      enable_research: request.enableResearch ?? false,
    }),
    timeoutMs: request.timeoutMs ?? DEFAULT_GENERATION_TIMEOUT_MS,
  });
};

export interface AcademicGenerationJob {
  project_id?: string;
  projectId?: string;
  job_id?: string | null;
  jobId?: string | null;
  id?: string | null;
  state?: string;
  status?: string;
  artifact_id?: string | null;
  artifactId?: string | null;
  poll_url?: string;
  output?: Record<string, unknown>;
  [key: string]: unknown;
}

export interface AcademicTypedResult {
  project_id?: string;
  projectId?: string;
  state: "ready" | "not_generated";
  artifact_id?: string | null;
  deck_id?: string | null;
  quiz_id?: string | null;
  deck?: unknown;
  questions?: unknown[];
  plan?: unknown;
  metrics?: unknown;
  artifact?: Record<string, unknown>;
  [key: string]: unknown;
}

export interface NotesGenerationConfig {
  mode?: "quick" | "detailed" | "exam" | "memory" | "complete";
  detail?: "concise" | "balanced" | "comprehensive";
  web?: boolean;
  exam_focus?: boolean;
  memory_aids?: boolean;
  scope?: "entire" | "selected";
  selected_topic_ids?: string[];
}

export interface NotesRecommendationRead {
  projectId: string;
  state: "ready" | "not_generated";
  recommendation?: {
    recommended_mode?: string;
    recommended_detail?: string;
    reason?: string;
    advisory?: boolean;
  };
  config?: Partial<NotesGenerationConfig>;
}

export const readNotesRecommendation = async (
  projectId: string
): Promise<NotesRecommendationRead> => {
  return apiFetch<NotesRecommendationRead>(
    `/api/academic/notes/recommendation/${encodeURIComponent(projectId)}`
  );
};

export const generateHierarchicalNotes = async (request: {
  projectId: string;
  documentIds?: string[];
  threshold?: number;
  config?: NotesGenerationConfig;
  timeoutMs?: number;
}): Promise<AcademicGenerationJob> => {
  const body: Record<string, unknown> = {
    project_id: request.projectId,
    document_ids: request.documentIds ?? [],
    threshold: request.threshold ?? 0.7,
  };
  if (request.config !== undefined) {
    body.config = request.config;
  }
  return apiFetch<AcademicGenerationJob>("/api/generation/notes/hierarchical", {
    method: "POST",
    body: JSON.stringify(body),
    timeoutMs: request.timeoutMs ?? DEFAULT_GENERATION_TIMEOUT_MS,
  });
};

/** Run the academic job to a terminal state for the Study Generate CTA. */
export const runHierarchicalNotes = async (request: {
  projectId: string;
  documentIds?: string[];
  threshold?: number;
  config?: NotesGenerationConfig;
  timeoutMs?: number;
  onProgress?: (job: Job) => void;
}): Promise<AcademicGenerationJob> => {
  const queued = await generateHierarchicalNotes(request);
  const jobId = queued.job_id || queued.jobId || queued.id;
  if (!jobId) return queued;
  const job = await pollJobUntilDone(jobId, {
    timeoutMs: request.timeoutMs ?? DEFAULT_JOB_POLL_TIMEOUT_MS,
    onUpdate: request.onProgress,
  });
  if (job.status !== "completed") {
    throw new JobRunError(
      `Notes job ${jobId} ${job.status}: ${job.errorMessage || "no failure detail was recorded"}.`,
      jobId,
      job
    );
  }
  return {
    ...queued,
    state: "ready",
    status: "completed",
    artifact_id:
      queued.artifact_id ||
      queued.artifactId ||
      (typeof job.output_data === "object" && job.output_data !== null
        ? String((job.output_data as Record<string, unknown>).artifact_id ?? "")
        : ""),
  };
};

/** Compatibility fallback for deployments before the academic POST exists. */
export const generateStudyNotes = async (request: {
  projectId: string;
  config?: NotesGenerationConfig;
  timeoutMs?: number;
  onProgress?: (job: Job) => void;
}): Promise<GeneratedContent | AcademicGenerationJob> => {
  try {
    return await runHierarchicalNotes(request);
  } catch (error) {
    const status = (error as { status?: number })?.status;
    if (status === 404 || status === 405 || status === 501) {
      return generateNotes({
        projectId: request.projectId,
        timeoutMs: request.timeoutMs,
      });
    }
    throw error;
  }
};

/**
 * Read the notes artifact the hierarchical job produced.
 *
 * The POST response is only a job handle, so callers that need the notes
 * themselves (e.g. the Website builder) read the artifact back: it lives in
 * the job output under `academic.notes_artifact` once synthesis ran. Returns
 * null when there is no artifact yet instead of inventing content.
 */
export const readHierarchicalNotes = async (
  projectId: string
): Promise<Record<string, unknown> | null> => {
  const read = await apiFetch<{ output?: Record<string, unknown> }>(
    `/api/generation/notes/hierarchical/${encodeURIComponent(projectId)}`
  );
  const output = read.output ?? {};
  const academic = (output.academic ?? {}) as Record<string, unknown>;
  const artifact = academic.notes_artifact;
  if (artifact && typeof artifact === "object") {
    return artifact as Record<string, unknown>;
  }
  return null;
};

export interface HierarchicalNotesGap {
  topic?: string;
  reason?: string;
}

/** Partial-success read for the notes run: the artifact plus honest gaps. */
export interface HierarchicalNotesRun {
  state: "ready" | "processing" | "failed" | "cancelled" | "not_generated";
  artifact_id?: string | null;
  artifact?: Record<string, unknown> | null;
  gaps: HierarchicalNotesGap[];
  failed_topic_count: number;
  provider_slow: boolean;
}

export const readHierarchicalNotesRun = async (
  projectId: string
): Promise<HierarchicalNotesRun> => {
  const read = await apiFetch<{
    state?: string;
    artifact_id?: string | null;
    gaps?: HierarchicalNotesGap[];
    failed_topic_count?: number;
    provider_slow?: boolean;
    output?: Record<string, unknown>;
  }>(`/api/generation/notes/hierarchical/${encodeURIComponent(projectId)}`);
  const output = (read.output ?? {}) as Record<string, unknown>;
  const academic = (output.academic ?? {}) as Record<string, unknown>;
  const artifact = academic.notes_artifact;
  return {
    state: (read.state as HierarchicalNotesRun["state"]) ?? "not_generated",
    artifact_id: read.artifact_id ?? null,
    artifact:
      artifact && typeof artifact === "object"
        ? (artifact as Record<string, unknown>)
        : null,
    gaps: Array.isArray(read.gaps) ? read.gaps : [],
    failed_topic_count:
      typeof read.failed_topic_count === "number" ? read.failed_topic_count : 0,
    provider_slow: Boolean(read.provider_slow),
  };
};

export const generateTypedFlashcards = async (request: {
  projectId: string;
  documentIds?: string[];
  maxCards?: number;
}): Promise<AcademicTypedResult> => {
  return apiFetch<AcademicTypedResult>("/api/generation/flashcards/typed", {
    method: "POST",
    body: JSON.stringify({
      project_id: request.projectId,
      document_ids: request.documentIds ?? [],
      max_cards: request.maxCards ?? null,
    }),
  });
};

export const generateMCQMatrix = async (request: {
  projectId: string;
  documentIds?: string[];
  total?: number;
}): Promise<AcademicTypedResult> => {
  return apiFetch<AcademicTypedResult>("/api/generation/mcq/matrix", {
    method: "POST",
    body: JSON.stringify({
      project_id: request.projectId,
      document_ids: request.documentIds ?? [],
      total: request.total ?? null,
    }),
    timeoutMs: PRACTICE_GENERATION_TIMEOUT_MS,
  });
};

export const generateMetrics = async (request: {
  projectId: string;
  documentIds?: string[];
  currentYear?: number;
  halfLife?: number;
  includeNarrative?: boolean;
}): Promise<AcademicTypedResult> => {
  return apiFetch<AcademicTypedResult>("/api/generation/metrics", {
    method: "POST",
    body: JSON.stringify({
      project_id: request.projectId,
      document_ids: request.documentIds ?? [],
      current_year: request.currentYear ?? null,
      half_life: request.halfLife ?? 5,
      include_narrative: request.includeNarrative ?? false,
    }),
  });
};

export const generateTypedFlashcardsWithFallback = async (request: {
  projectId: string;
}): Promise<GeneratedContent | AcademicTypedResult> => {
  try {
    return await generateTypedFlashcards(request);
  } catch (error) {
    const status = (error as { status?: number })?.status;
    if (status === 404 || status === 405 || status === 501) {
      return generateFlashcards({ projectId: request.projectId });
    }
    throw error;
  }
};

export const generateMCQMatrixWithFallback = async (request: {
  projectId: string;
}): Promise<GeneratedContent | AcademicTypedResult> => {
  try {
    return await generateMCQMatrix(request);
  } catch (error) {
    const status = (error as { status?: number })?.status;
    if (status === 404 || status === 405 || status === 501) {
      return generateMCQ({ projectId: request.projectId });
    }
    throw error;
  }
};

export const generateFlashcards = async (request: GenerateFlashcardsRequest): Promise<GeneratedContent> => {
  return apiFetch<GeneratedContent>("/api/generation/flashcards", {
    method: "POST",
    body: JSON.stringify({
      project_id: request.projectId,
      query: request.query ?? null,
      enable_research: request.enableResearch ?? false,
    }),
    timeoutMs: request.timeoutMs ?? DEFAULT_GENERATION_TIMEOUT_MS,
  });
};

export const generateQuiz = async (request: GenerateQuizRequest): Promise<GeneratedContent> => {
  return apiFetch<GeneratedContent>("/api/generation/quiz", {
    method: "POST",
    body: JSON.stringify({
      project_id: request.projectId,
      query: request.query ?? null,
      enable_research: request.enableResearch ?? false,
    }),
    timeoutMs: request.timeoutMs ?? DEFAULT_GENERATION_TIMEOUT_MS,
  });
};

export const generateSummary = async (request: GenerateSummaryRequest): Promise<GeneratedContent> => {
  return apiFetch<GeneratedContent>("/api/generation/summary", {
    method: "POST",
    body: JSON.stringify({
      project_id: request.projectId,
      query: request.query ?? null,
      enable_research: request.enableResearch ?? false,
    }),
  });
};

export const generateWebsite = async (request: GenerateWebsiteRequest): Promise<GeneratedContent> => {
  return apiFetch<GeneratedContent>("/api/generation/website", {
    method: "POST",
    body: JSON.stringify({
      project_id: request.projectId,
      query: request.query ?? null,
      enable_research: request.enableResearch ?? false,
    }),
  });
};

export interface GenerateAnswerRequest {
  projectId: string;
  query: string;
  marks?: number;
  style?: "concise" | "detailed" | "exam" | "tutorial";
  enableResearch?: boolean;
}

export interface AnswerResult {
  projectId: string;
  mode: string;
  content: {
    question?: string;
    answer_text?: string;
    marks?: number;
    citations?: Citation[];
    provenance?: { support?: string; note?: string };
    [key: string]: unknown;
  };
  cached: boolean;
  processingTime: number;
  marks: number;
  word_band: { min_words: number; max_words: number };
}

export const generateAnswer = async (request: GenerateAnswerRequest): Promise<AnswerResult> => {
  if (!(request.query || "").trim()) {
    throw new Error("Query is required for answer generation.");
  }
  return apiFetch<AnswerResult>("/api/generation/answer", {
    method: "POST",
    body: JSON.stringify({
      project_id: request.projectId,
      query: request.query,
      marks: request.marks ?? 5,
      style: request.style ?? "detailed",
      enable_research: request.enableResearch ?? false,
    }),
  });
};

export interface GenerateExplanationRequest {
  projectId: string;
  query: string;
  enableResearch?: boolean;
  timeoutMs?: number;
}

export interface ExplanationResult {
  projectId: string;
  mode: string;
  content: {
    explanation?: string;
    examples?: string[];
    related_concepts?: string[];
    citations?: Citation[];
    provenance?: { support?: string; note?: string };
    [key: string]: unknown;
  };
  cached: boolean;
  processingTime: number;
}

export const generateExplanation = async (
  request: GenerateExplanationRequest
): Promise<ExplanationResult> => {
  if (!(request.query || "").trim()) {
    throw new Error("Query is required for explanation generation.");
  }
  return apiFetch<ExplanationResult>("/api/generation/explanation", {
    method: "POST",
    body: JSON.stringify({
      project_id: request.projectId,
      query: request.query,
      enable_research: request.enableResearch ?? false,
    }),
    timeoutMs: request.timeoutMs ?? DEFAULT_GENERATION_TIMEOUT_MS,
  });
};

export const generateMCQ = async (request: GenerateQuizRequest): Promise<GeneratedContent> => {
  return apiFetch<GeneratedContent>("/api/generation/mcq", {
    method: "POST",
    body: JSON.stringify({
      project_id: request.projectId,
      query: request.query ?? null,
      enable_research: request.enableResearch ?? false,
    }),
    timeoutMs: request.timeoutMs ?? PRACTICE_GENERATION_TIMEOUT_MS,
  });
};

// Single Practice generator (Phase B: one Practice surface). Strict MCQs
// (exactly 4 distinct options per question with explanations) back the
// whole Practice tab; the legacy quiz endpoint stays for compatibility.
export const generatePractice = async (request: GenerateQuizRequest): Promise<GeneratedContent> => {
  return generateMCQ(request);
};

// ============================================================
// Notes / flashcards / quiz reads (ownership-scoped GETs)
// ============================================================

export interface NoteSection {
  heading: string;
  content: string;
  subsections?: NoteSection[];
  citations?: Citation[];
}

/**
 * One row of the notes LIST. The backend returns the whole notes artifact
 * set (newest first) with the metadata the list needs: id, title, created,
 * mode, coverage, pages, and the PDF artifact identity used by Open/Delete.
 */
export interface NotesListItem {
  id: string;
  projectId: string;
  title: string;
  content: unknown;
  createdAt: string | null;
  created_at?: string | null;
  mode?: string | null;
  detail?: string | null;
  version?: number | null;
  pages?: number | null;
  coverage?: {
    topic_coverage?: number | null;
    concept_coverage?: number | null;
    passes?: boolean | null;
    threshold?: number | null;
    uncovered?: string[];
  };
  topics?: number | null;
  sections?: number | null;
  artifactId?: string | null;
  status?: string;
  model?: string | null;
}

export interface NotesRead {
  projectId: string;
  state: "ready" | "not_generated";
  message?: string;
  notes: NotesListItem[];
  total: number;
}

export const readNotes = async (projectId: string): Promise<NotesRead> => {
  return apiFetch<NotesRead>(`/api/generation/notes/${projectId}`);
};

/**
 * Delete one note of this project (server enforces ownership: another user's
 * note id is a 404). The note's stored PDF goes with it.
 */
export const deleteNote = async (
  projectId: string,
  noteId: string
): Promise<{ deleted: boolean; noteId: string; artifactId?: string | null }> => {
  return apiFetch<{ deleted: boolean; noteId: string; artifactId?: string | null }>(
    `/api/generation/notes/${encodeURIComponent(projectId)}/${encodeURIComponent(noteId)}`,
    { method: "DELETE" }
  );
};

// ============================================================
// Study Notes PDF + Insights (academic read paths, Step 8-10)
//
// The study-notes PDF is rendered server-side from the structured notes
// artifact and streamed as bytes; the EXISTING DocumentViewer renders it
// (no second viewer). Insights serves backend-first summary cards and
// topic table rows. Empty projects return honest not_generated states.
// ============================================================

export interface StudyNotesCoverage {
  topic_coverage?: number | null;
  concept_coverage?: number | null;
  passes?: boolean | null;
  threshold?: number | null;
  uncovered?: string[];
}

export interface StudyNotesArtifactSummary {
  id?: string;
  artifact_id?: string;
  artifactId?: string;
  artifact_type?: string;
  title?: string;
  version?: number | null;
  status?: string;
  pages?: number | null;
  topics?: number | null;
  sections?: number | null;
  mode?: string | null;
  detail?: string | null;
  coverage?: StudyNotesCoverage;
  model?: string;
  sha256?: string;
  created_at?: string;
  document?: string;
}

export interface StudyNotesPdfMeta {
  projectId: string;
  state: "ready" | "not_generated";
  kind?: string;
  message?: string;
  generation_call?: string;
  title?: string;
  artifact_id?: string;
  version?: number;
  status?: string;
  model_version?: number | null;
  topics?: number | null;
  sections?: number | null;
  pages?: number | null;
  coverage?: StudyNotesCoverage;
  model?: string;
  sha256?: string;
  created_at?: string;
  document?: string;
  /** Every stored version, newest first (drives the notes list). */
  artifacts?: StudyNotesArtifactSummary[];
  total?: number;
}

const studyNotesVersionQuery = (version?: number | null): string =>
  version && version > 0 ? `?version=${encodeURIComponent(String(version))}` : "";

export const readStudyNotesPdfMeta = async (
  projectId: string,
  version?: number | null
): Promise<StudyNotesPdfMeta> => {
  return apiFetch<StudyNotesPdfMeta>(
    `/api/academic/notes/${encodeURIComponent(projectId)}${studyNotesVersionQuery(version)}`
  );
};

export const fetchStudyNotesPdf = async (
  projectId: string,
  version?: number | null
): Promise<Blob> => {
  // Stored PDF bytes for the viewer (GET /api/academic/notes/{id}/pdf,
  // JWT ownership enforced server-side). Fetched as a Blob so the viewer
  // can render without ever exposing the token in a URL. An optional version
  // opens THAT stored version instead of only the newest one.
  const token = getAuthToken();
  const response = await fetch(
    `${API_BASE_URL}/api/academic/notes/${encodeURIComponent(projectId)}/pdf${studyNotesVersionQuery(version)}`,
    { headers: token ? { Authorization: `Bearer ${token}` } : {} }
  );
  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: "Study notes unavailable" }));
    const httpError = new Error(error.detail || "Study notes unavailable") as Error & {
      status?: number;
    };
    httpError.status = response.status;
    throw httpError;
  }
  return response.blob();
};

export interface InsightCard {
  card: string;
  [key: string]: unknown;
}

export interface InsightTableRow {
  topic: string;
  unit: string;
  frequency: number;
  recurrence: number;
  marks_total: number;
  marks_share: number;
  recency_score: number;
  hotspot_rank: number;
  coverage: { syllabus: boolean; documents: boolean; pyq: boolean; questions: boolean };
  evidence: Record<string, string[]>;
}

export interface InsightsRead {
  projectId: string;
  state: "ready" | "not_generated";
  kind?: string;
  message?: string;
  generation_call?: string;
  cards: InsightCard[];
  table: InsightTableRow[];
  never_asked: string[];
  missing: string[];
  totals: Record<string, unknown>;
  narrative?: string | null;
  narrative_status?: string;
  provenance_note?: string;
}

export const readInsights = async (projectId: string): Promise<InsightsRead> => {
  return apiFetch<InsightsRead>(`/api/academic/insights/${projectId}`);
};

export const readTypedFlashcards = async (projectId: string): Promise<AcademicTypedResult> => {
  return apiFetch<AcademicTypedResult>(`/api/generation/flashcards/typed/${projectId}`);
};

export const readMCQMatrix = async (projectId: string): Promise<AcademicTypedResult> => {
  return apiFetch<AcademicTypedResult>(`/api/generation/mcq/matrix/${projectId}`);
};

export const readMetrics = async (projectId: string): Promise<AcademicTypedResult> => {
  return apiFetch<AcademicTypedResult>(`/api/generation/metrics/${projectId}`);
};

export interface QuizRead {
  projectId: string;
  state: "ready" | "not_generated";
  message?: string;
  quizzes: Array<{ id: string; projectId: string; title: string; questions: unknown[]; createdAt: string | null }>;
  total: number;
}

export const readQuiz = async (projectId: string): Promise<QuizRead> => {
  return apiFetch<QuizRead>(`/api/generation/quiz/${projectId}`);
};

// ============================================================
// Academic Insights (syllabus / PYQ / mapping / analysis)
// ============================================================

export interface AcademicCitation {
  chunk_id: string;
  document_id?: string | null;
  page?: number | null;
  span?: string;
}

export interface SyllabusUnit {
  unit_id: string;
  title: string;
  topics: string[];
  outcomes?: string[];
  weight_percent?: number | null;
  citations?: AcademicCitation[];
}

export interface SyllabusContent {
  title: string;
  course?: string | null;
  units: SyllabusUnit[];
  citations?: AcademicCitation[];
  provenance?: { support?: string; note?: string };
}

export interface PYQItem {
  question: string;
  year: number;
  marks: number;
  unit_id?: string | null;
  topic?: string | null;
  paper?: string | null;
  citations?: AcademicCitation[];
}

export interface PYQContent {
  title: string;
  questions: PYQItem[];
  citations?: AcademicCitation[];
  provenance?: { support?: string; note?: string };
}

export interface UnitMapping {
  question_index: number;
  question: string;
  unit_id: string | null;
  unit_title: string | null;
  confidence: number;
  method: "token_overlap" | "explicit_unit" | "unmapped";
}

export interface MappingContent {
  syllabus_title: string;
  mappings: UnitMapping[];
  unmapped_count: number;
  coverage_pct: number;
}

export interface UnitFrequency {
  unit_id: string;
  unit_title: string;
  count: number;
  marks_total: number;
  share_pct: number;
}

export interface AnalysisContent {
  syllabus_title: string;
  total_questions: number;
  mapped_questions: number;
  unmapped_questions: number;
  frequency: UnitFrequency[];
  total_marks: number;
  coverage_pct: number;
}

export interface AcademicResult<T> {
  project_id: string;
  mode: string;
  content: T;
  cached: boolean;
  processing_time: number;
}

export interface AcademicRead<T> {
  projectId: string;
  state: "ready" | "not_generated";
  message?: string;
  total: number;
  [key: string]: unknown;
}

export const generateSyllabus = async (request: {
  projectId: string;
  query?: string;
}): Promise<AcademicResult<SyllabusContent>> => {
  return apiFetch<AcademicResult<SyllabusContent>>("/api/generation/syllabus", {
    method: "POST",
    body: JSON.stringify({ project_id: request.projectId, query: request.query ?? null }),
  });
};

export const generatePyq = async (request: {
  projectId: string;
  query?: string;
}): Promise<AcademicResult<PYQContent>> => {
  return apiFetch<AcademicResult<PYQContent>>("/api/generation/pyq", {
    method: "POST",
    body: JSON.stringify({ project_id: request.projectId, query: request.query ?? null }),
  });
};

export const generateMapping = async (request: {
  projectId: string;
}): Promise<AcademicResult<MappingContent>> => {
  return apiFetch<AcademicResult<MappingContent>>("/api/generation/mapping", {
    method: "POST",
    body: JSON.stringify({ project_id: request.projectId }),
  });
};

export const generateAnalysis = async (request: {
  projectId: string;
}): Promise<AcademicResult<AnalysisContent>> => {
  return apiFetch<AcademicResult<AnalysisContent>>("/api/generation/analysis", {
    method: "POST",
    body: JSON.stringify({ project_id: request.projectId }),
  });
};

export const readSyllabus = async (projectId: string): Promise<AcademicRead<SyllabusContent>> => {
  return apiFetch<AcademicRead<SyllabusContent>>(`/api/generation/syllabus/${projectId}`);
};

export const readPyq = async (projectId: string): Promise<AcademicRead<PYQContent>> => {
  return apiFetch<AcademicRead<PYQContent>>(`/api/generation/pyq/${projectId}`);
};

export const readMapping = async (projectId: string): Promise<AcademicRead<MappingContent>> => {
  return apiFetch<AcademicRead<MappingContent>>(`/api/generation/mapping/${projectId}`);
};

export const readAnalysis = async (projectId: string): Promise<AcademicRead<AnalysisContent>> => {
  return apiFetch<AcademicRead<AnalysisContent>>(`/api/generation/analysis/${projectId}`);
};

// ============================================================
// Study Website pipeline (content -> assembly -> sanitise ->
// preview -> publish -> public route)
// ============================================================

export type WebsiteContentType = "notes" | "flashcards" | "quiz" | "mixed";

export interface StudyWebsite {
  id: string;
  project_id: string;
  slug: string;
  content_type: WebsiteContentType;
  title: string;
  description?: string | null;
  is_published: boolean;
  status: "draft" | "published";
  visibility: "private" | "public";
  created_at: string;
  updated_at: string;
}

export interface WebsiteGenerationResult {
  website_id: string;
  slug: string;
  title: string;
  content_type: WebsiteContentType;
  html_content: string;
  preview_url: string;
  created_at: string;
}

export interface CreateStudyWebsiteRequest {
  projectId: string;
  contentType: WebsiteContentType;
  title?: string;
  description?: string;
  // Included resources: structured content assembled from
  // notes/flashcards/MCQs/explanations/answers generation output.
  structuredContent: Record<string, unknown>;
  /** Client-side deadline for the whole POST (default 240s). */
  timeoutMs?: number;
}

export const createStudyWebsite = async (
  request: CreateStudyWebsiteRequest
): Promise<WebsiteGenerationResult> => {
  // Correct endpoint: POST /api/website/generate with
  // content_type/title/structured_content (NOT /api/generation/website,
  // which only returns structured content for a study page).
  return apiFetch<WebsiteGenerationResult>("/api/website/generate", {
    method: "POST",
    body: JSON.stringify({
      project_id: request.projectId,
      content_type: request.contentType,
      title: request.title,
      description: request.description,
      structured_content: request.structuredContent,
    }),
    timeoutMs: request.timeoutMs ?? DEFAULT_WEBSITE_TIMEOUT_MS,
  });
};

export const listWebsites = async (projectId: string): Promise<StudyWebsite[]> => {
  return apiFetch<StudyWebsite[]>(`/api/website/list/${projectId}`);
};

export const getWebsiteDetail = async (slug: string): Promise<StudyWebsite> => {
  return apiFetch<StudyWebsite>(`/api/website/detail/${slug}`);
};

export const previewWebsiteHtml = async (slug: string): Promise<string> => {
  // Persisted sanitised HTML (same bytes the public route serves).
  return apiFetch<string>(`/api/website/raw/${slug}`, {}).then(
    (body) => (body as unknown as { html_content: string }).html_content
  );
};

export const publishWebsite = async (slug: string): Promise<{ status: string; slug: string; preview_url: string; public_url: string }> => {
  return apiFetch(`/api/website/publish`, {
    method: "POST",
    body: JSON.stringify({ slug }),
  });
};

export const unpublishWebsite = async (slug: string): Promise<{ status: string; slug: string }> => {
  return apiFetch(`/api/website/unpublish`, {
    method: "POST",
    body: JSON.stringify({ slug }),
  });
};

export const setWebsiteVisibility = async (
  slug: string,
  visibility: "public" | "private"
): Promise<{ status: string; slug: string; visibility: string }> => {
  return apiFetch(`/api/website/visibility`, {
    method: "POST",
    body: JSON.stringify({ slug, visibility }),
  });
};

export const updateWebsite = async (
  slug: string,
  patch: { title?: string; description?: string; visibility?: "public" | "private" }
): Promise<StudyWebsite> => {
  return apiFetch<StudyWebsite>(`/api/website/manage/${slug}`, {
    method: "PATCH",
    body: JSON.stringify(patch),
  });
};

export const deleteWebsite = async (slug: string): Promise<boolean> => {
  await apiFetch(`/api/website/manage/${slug}`, { method: "DELETE" });
  return true;
};

export const resolvePublicWebsite = async (slug: string): Promise<StudyWebsite> => {
  // Domain-agnostic public resolve: published only, never leaks private docs.
  return apiFetch<StudyWebsite>(`/api/website/public/${slug}`);
};

export const publicWebsiteUrl = (slug: string): string => {
  return `${API_BASE_URL}/api/website/public/${slug}/html`;
};

// ============================================================
// Study Session API
// ============================================================

interface StartSessionRequest {
  projectId: string;
  sessionType: "flashcards" | "quiz" | "notes" | "tutor";
  limit?: number;
}

interface SubmitResponseRequest {
  sessionId: string;
  itemId: string;
  response: string | number;
}

interface CompleteSessionRequest {
  sessionId: string;
}

export const startStudySession = async (request: StartSessionRequest): Promise<StudySession> => {
  // Snake_case wire contract; user identity comes from the server JWT
  // (never a client-supplied user_id).
  return apiFetch<StudySession>("/api/study/session/start", {
    method: "POST",
    body: JSON.stringify({
      project_id: request.projectId,
      session_type: request.sessionType,
      ...(request.limit !== undefined ? { limit: request.limit } : {}),
    }),
  });
};

export const getCurrentStudyItem = async (sessionId: string): Promise<StudyItem> => {
  return apiFetch<StudyItem>(`/api/study/session/${sessionId}/current`);
};

export const submitStudyResponse = async (request: SubmitResponseRequest): Promise<StudyResponse> => {
  return apiFetch<StudyResponse>("/api/study/session/respond", {
    method: "POST",
    body: JSON.stringify({
      session_id: request.sessionId,
      item_id: request.itemId,
      response: request.response,
    }),
  });
};

export const completeStudySession = async (request: CompleteSessionRequest): Promise<SessionResult> => {
  return apiFetch<SessionResult>("/api/study/session/complete", {
    method: "POST",
    body: JSON.stringify({ session_id: request.sessionId }),
  });
};

export const getSessionAnalytics = async (sessionId: string): Promise<unknown> => {
  return apiFetch(`/api/study/session/${sessionId}/analytics`);
};

export const getStudyHistory = async (userId: string): Promise<unknown> => {
  return apiFetch(`/api/study/history/${userId}`);
};

// ============================================================
// Retrieval API (for AI Tutor)
// ============================================================

interface RetrievalRequest {
  query: string;
  projectId?: string;
  documentId?: string;
  topK?: number;
  minSimilarity?: number;
}

export const searchContent = async (request: RetrievalRequest) => {
  // Single snake_case wire convention (backend also accepts camelCase
  // aliases, but the frontend always sends snake_case).
  return apiFetch("/api/retrieval/search", {
    method: "POST",
    body: JSON.stringify({
      query: request.query,
      ...(request.projectId !== undefined ? { project_id: request.projectId } : {}),
      ...(request.documentId !== undefined ? { document_id: request.documentId } : {}),
      ...(request.topK !== undefined ? { top_k: request.topK } : {}),
      ...(request.minSimilarity !== undefined ? { min_similarity: request.minSimilarity } : {}),
    }),
  });
};

// ============================================================
// AI Tutor API
// ============================================================

interface TutorMessageRequest {
  sessionId: string;
  message: string;
}

export const sendTutorMessage = async (request: TutorMessageRequest): Promise<TutorResponse> => {
  return apiFetch<TutorResponse>("/api/study/tutor/message", {
    method: "POST",
    body: JSON.stringify({ session_id: request.sessionId, message: request.message }),
  });
};

// ============================================================
// Auth helpers
// ============================================================

export const setAuthToken = (token: string) => {
  localStorage.setItem(ACCESS_TOKEN_KEY, token);
};

export const clearAuthToken = () => {
  localStorage.removeItem(ACCESS_TOKEN_KEY);
};

export const login = async (email: string, password: string): Promise<AuthResponse> => {
  // Normalise the email the same way the backend does (trim + lowercase)
  // so mixed-case/padded input always hits the same account.
  const normalizedEmail = email.trim().toLowerCase();
  let response: {
    access_token: string;
    refresh_token?: string;
    user: User;
  };
  try {
    response = await apiFetch<{
      access_token: string;
      refresh_token?: string;
      user: User;
    }>("/api/auth/login", {
      method: "POST",
      body: JSON.stringify({ email: normalizedEmail, password }),
    });
  } catch (err) {
    // Rejected credentials must not strand a previous session.
    clearStoredAuthSession();
    throw err;
  }
  clearStoredAuthSession();
  setAuthToken(response.access_token);
  if (response.refresh_token) {
    localStorage.setItem(REFRESH_TOKEN_KEY, response.refresh_token);
  }
  localStorage.setItem(USER_CACHE_KEY, JSON.stringify(response.user));
  return { user: response.user, token: response.access_token };
};

export const register = async (email: string, password: string, name: string): Promise<AuthResponse> => {
  // Same normalisation as login: trim + lowercase before send.
  const normalizedEmail = email.trim().toLowerCase();
  let response: {
    access_token: string;
    refresh_token?: string;
    user: User;
  };
  try {
    response = await apiFetch<{
      access_token: string;
      refresh_token?: string;
      user: User;
    }>("/api/auth/register", {
      method: "POST",
      body: JSON.stringify({ email: normalizedEmail, password, name }),
    });
  } catch (err) {
    clearStoredAuthSession();
    throw err;
  }
  clearStoredAuthSession();
  setAuthToken(response.access_token);
  if (response.refresh_token) {
    localStorage.setItem(REFRESH_TOKEN_KEY, response.refresh_token);
  }
  localStorage.setItem(USER_CACHE_KEY, JSON.stringify(response.user));
  return { user: response.user, token: response.access_token };
};

export const logout = async (): Promise<void> => {
  try {
    await apiFetch("/api/auth/logout", { method: "POST" });
  } catch {
    // Server logout is best-effort (revokes all tokens); local session
    // is cleared regardless so a failed call cannot strand the user.
  }
  clearStoredAuthSession();
};

export const getCurrentUser = async (): Promise<User | null> => {
  if (!getAuthToken()) return null;

  try {
    const user = await apiFetch<User>("/api/auth/me");
    localStorage.setItem(USER_CACHE_KEY, JSON.stringify(user));
    return user;
  } catch (err) {
    const status = (err as { status?: number })?.status;
    // apiFetch has already attempted one silent refresh. Its helper clears
    // local auth only when that refresh itself fails. A second 401 keeps the
    // renewed tokens and is returned as signed-out without an auth loop.
    if (status === 401) return null;

    // Network/5xx failures must not wipe a valid session. Keep the cached user
    // until the next stale-window revalidation.
    try {
      const cached = localStorage.getItem(USER_CACHE_KEY);
      if (cached) return JSON.parse(cached) as User;
    } catch {
      // Corrupt cache: fall through to null without clearing the token.
    }
    return null;
  }
};

let refreshInFlight: Promise<string | null> | null = null;

/** Exchange the stored refresh token once; concurrent 401s share this promise. */
export const refreshAccessToken = async (
  staleAccessToken?: string
): Promise<string | null> => {
  if (typeof window === "undefined") return null;

  const currentAccess = getAuthToken();
  if (staleAccessToken && currentAccess && currentAccess !== staleAccessToken) {
    return currentAccess;
  }
  if (refreshInFlight) return refreshInFlight;

  const refreshToken = localStorage.getItem(REFRESH_TOKEN_KEY);
  if (!refreshToken) {
    clearStoredAuthSession();
    return null;
  }

  const pending = (async (): Promise<string | null> => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/auth/refresh`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ refresh_token: refreshToken }),
        credentials: "include",
      });
      if (!response.ok) {
        clearStoredAuthSession();
        return null;
      }
      const data = (await response.json().catch(() => null)) as {
        access_token?: string;
        refresh_token?: string;
      } | null;
      const access = data?.access_token;
      if (!access) {
        clearStoredAuthSession();
        return null;
      }
      setAuthToken(access);
      if (data?.refresh_token) {
        localStorage.setItem(REFRESH_TOKEN_KEY, data.refresh_token);
      }
      return access;
    } catch {
      clearStoredAuthSession();
      return null;
    }
  })();

  refreshInFlight = pending;
  try {
    return await pending;
  } finally {
    if (refreshInFlight === pending) refreshInFlight = null;
  }
};

// ============================================================
// Flashcard API
// ============================================================

interface GetFlashcardsRequest {
  projectId: string;
}

interface GetFlashcardDecksRequest {
  projectId: string;
}

export const getFlashcards = async (request: GetFlashcardsRequest): Promise<FlashcardDeck> => {
  // Call the real API to get flashcards
  try {
    const response = await apiFetch<FlashcardDeck>(`/api/generation/flashcards/${request.projectId}`);
    console.log("[API] Flashcards fetched:", response);
    return response;
  } catch (error) {
    console.error("[API Error] Failed to fetch flashcards:", error);
    // Return empty deck - let UI show empty state
    return {
      id: "deck_empty",
      projectId: request.projectId,
      name: "No Flashcards",
      cards: [],
      cardCount: 0,
    };
  }
};

export const getFlashcardDecks = async (request: GetFlashcardDecksRequest): Promise<FlashcardDeck[]> => {
  // Call the real API to get flashcard decks
  try {
    const decks = await apiFetch<FlashcardDeck[]>(`/api/projects/${request.projectId}/flashcard-decks`);
    console.log("[API] Flashcard decks fetched:", decks);
    return decks;
  } catch (error) {
    console.error("[API Error] Failed to fetch flashcard decks:", error);
    // Return empty array - let UI show empty state
    return [];
  }
};

// ============================================================
// AI Provider settings (Settings > AI Provider)
//
// User-owned LLM keys (Groq / OpenRouter). Raw keys travel inbound in
// POST/PUT bodies only; every response carries masked hints, never raw
// keys. All bodies are snake_case per the single wire convention.
// ============================================================

export type AIProviderName = "groq" | "openrouter";

export interface AIProviderEntry {
  provider: AIProviderName;
  configured: boolean;
  key_hint_masked: string | null;
  model: string | null;
  source: "user" | "app";
  effective_model: string;
  app_model: string;
  app_key_available: boolean;
}

export interface AIProviderStatus {
  active_provider: string;
  providers: AIProviderEntry[];
  effective: {
    provider: string;
    model: string;
    source: "project" | "user" | "app";
    using_user_key: boolean;
    key_hint_masked: string | null;
  };
}

export interface AIModelEntry {
  id: string;
  label: string;
  /** OpenRouter display name (defaults to id when absent). */
  name?: string;
  /** Context window in tokens (OpenRouter live catalog). */
  context_length?: number;
  /** Human pricing hint, e.g. "free" or "$0.14/1M" (display only). */
  pricing?: string;
}

export const getProviderStatus = async (): Promise<AIProviderStatus> => {
  return apiFetch<AIProviderStatus>("/api/settings/providers/status");
};

export const listProviderModels = async (opts?: {
  provider?: AIProviderName;
  search?: string;
  /** Alias for search (GET /models?q=). */
  q?: string;
  /** Live provider lists (server-side fetch + 1h cache, static fallback). */
  live?: boolean;
  refresh?: boolean;
  limit?: number;
  /** AbortSignal from React Query so debounced keystrokes cancel in-flight fetches. */
  signal?: AbortSignal;
}): Promise<{ providers: string[]; models: Record<string, AIModelEntry[]>; source?: string }> => {
  const params = new URLSearchParams();
  if (opts?.provider) params.set("provider", opts.provider);
  if (opts?.search) params.set("search", opts.search);
  else if (opts?.q) params.set("q", opts.q);
  if (opts?.live) params.set("live", "true");
  if (opts?.refresh) params.set("refresh", "true");
  if (opts?.limit !== undefined) params.set("limit", String(opts.limit));
  const query = params.toString();
  return apiFetch(`/api/settings/providers/models${query ? `?${query}` : ""}`, {
    ...(opts?.signal ? { signal: opts.signal } : {}),
  });
};

export const saveProviderSettings = async (request: {
  provider: AIProviderName;
  apiKey: string;
  model: string;
}): Promise<{ saved: boolean; provider: string; key_hint_masked: string; model: string }> => {
  return apiFetch("/api/settings/providers", {
    method: "PUT",
    body: JSON.stringify({
      provider: request.provider,
      api_key: request.apiKey,
      model: request.model,
    }),
  });
};

export const removeProviderSettings = async (
  provider: AIProviderName
): Promise<{ removed: boolean; provider: string }> => {
  return apiFetch(`/api/settings/providers/${provider}`, { method: "DELETE" });
};

export interface ProviderTestResult {
  ok: boolean;
  provider: string;
  model: string;
  latency_ms: number;
  warning?: string;
}

export const testProviderConnection = async (request: {
  provider: AIProviderName;
  useSaved?: boolean;
  apiKey?: string;
  model?: string;
}): Promise<ProviderTestResult> => {
  return apiFetch<ProviderTestResult>("/api/settings/providers/test", {
    method: "POST",
    body: JSON.stringify({
      provider: request.provider,
      use_saved: request.useSaved ?? false,
      ...(request.apiKey !== undefined ? { api_key: request.apiKey } : {}),
      ...(request.model !== undefined ? { model: request.model } : {}),
    }),
  });
};

export const setProjectProviderOverride = async (
  projectId: string,
  provider: AIProviderName,
  model: string
): Promise<{ project_id: string; override: { provider: string; model: string } }> => {
  return apiFetch(`/api/settings/providers/project/${projectId}`, {
    method: "PUT",
    body: JSON.stringify({ provider, model }),
  });
};

export const clearProjectProviderOverride = async (
  projectId: string
): Promise<{ project_id: string; override: null }> => {
  return apiFetch(`/api/settings/providers/project/${projectId}`, { method: "DELETE" });
};

// ============================================================
// TTS API
// ============================================================

interface TTSRequest {
  text: string;
  voice?: string;
}

interface TTSResponse {
  success: boolean;
  voice: string;
  format: string;
  sampleRate: number;
  durationMs: number;
  processingTime: number;
}

export const synthesizeSpeech = async (request: TTSRequest): Promise<TTSResponse> => {
  return apiFetch<TTSResponse>("/api/tts/synthesize", {
    method: "POST",
    body: JSON.stringify(request),
  });
};

export const getTTSVoices = async () => {
  return apiFetch("/api/tts/voices");
};

// ============================================================
// Puter optional heavy-document path (default OFF)
// ============================================================

// Puter is an explicit, user-authorised fallback for heavy documents.
// Default OFF: these wrappers throw unless the caller passes an explicit
// gesture ({ authorised: true }) AND the feature flag is enabled.
// No credentials are stored: the caller supplies a short-lived user token
// per call, forwarded as a Bearer header and never persisted.

export const PUTER_ENABLED: boolean =
  (VITE_ENV.VITE_PUTER_ENABLED ?? "false") === "true";

export const PUTER_HEAVY_PAGE_THRESHOLD: number = Number(
  VITE_ENV.VITE_PUTER_HEAVY_PAGE_THRESHOLD ?? "100"
) || 100;

export interface PuterAuthorisation {
  /** Explicit user gesture (checkbox/button) — required, never implied. */
  authorised: boolean;
  /** Short-lived user token for this call only — never stored. */
  userToken?: string;
}

export const shouldWarnHeavyDocument = (pageCount: number): boolean => {
  // Configurable warning threshold (default ~100 pages), not a hard invariant.
  return pageCount >= PUTER_HEAVY_PAGE_THRESHOLD;
};

export const heavyDocumentWarning = (pageCount: number): string => {
  return (
    `This document has ~${pageCount} pages (threshold ~${PUTER_HEAVY_PAGE_THRESHOLD}). ` +
    `Local PP-OCR will process it page by page; for faster heavy-document OCR you may ` +
    `explicitly authorise the optional Puter path. Puter is OFF by default and any ` +
    `failure degrades explicitly with provenance preserved.`
  );
};

export const puterOcrPage = async (
  image: Blob,
  pageNumber: number,
  auth: PuterAuthorisation,
  opts?: { totalPages?: number }
): Promise<{ text: string; confidence: number; method: string }> => {
  if (!PUTER_ENABLED) {
    throw new Error("Puter path is disabled by default (set VITE_PUTER_ENABLED=true to offer it).");
  }
  if (!auth?.authorised || !auth?.userToken) {
    throw new Error("Puter requires an explicit user gesture + token per call; nothing is stored.");
  }
  const token = getAuthToken();
  const headers: Record<string, string> = {
    "Content-Type": "application/octet-stream",
    "X-Puter-Page": String(pageNumber),
    // Per-call Puter user token (never persisted, never logged server-side).
    "X-Puter-Token": auth.userToken,
  };
  if (opts?.totalPages !== undefined) {
    headers["X-Puter-Pages"] = String(opts.totalPages);
  }
  if (token) {
    // App JWT stays on Authorization so the backend ownership check runs.
    headers["Authorization"] = `Bearer ${token}`;
  }
  const response = await fetch(`${API_BASE_URL}/api/ocr/puter`, {
    method: "POST",
    headers,
    body: image,
  });
  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: "Puter OCR degraded" }));
    // Explicit degraded (never silent): surface the reason to the caller.
    throw new Error(error.detail || "Puter OCR degraded explicitly");
  }
  return response.json();
};

export interface PuterHeavyCheck {
  heavy: boolean;
  pageCount: number;
  threshold: number;
  enabled: boolean;
  warning: string | null;
}

export interface PuterProcessResult {
  documentId: string;
  jobId: string | null;
  stage: string;
  indexingState: string | null;
  totalPages: number;
  chunks: number;
  pageMethods: Record<string, number>;
  puterPages: number;
  fallbackPages: number;
  partial: boolean;
  error: string | null;
  message: string;
}

/** Heavy check API: is this document heavy (configurable threshold)? */
export const puterHeavyCheck = async (pageCount: number): Promise<PuterHeavyCheck> => {
  return apiFetch(`/api/ocr/puter/heavy-check?page_count=${encodeURIComponent(pageCount)}`);
};

/**
 * Authorised heavy-document run: Puter vision -> normal pipeline.
 * Requires the explicit gesture ({ authorised: true }) plus a short-lived
 * per-call user token (never persisted anywhere — local state only).
 * Deny/fail -> caller falls back to the normal local pipeline.
 */
export const puterProcessDocument = async (
  documentId: string,
  auth: PuterAuthorisation
): Promise<PuterProcessResult> => {
  if (!PUTER_ENABLED) {
    throw new Error("Puter path is disabled by default (set VITE_PUTER_ENABLED=true to offer it).");
  }
  if (!auth?.authorised || !auth?.userToken) {
    throw new Error("Puter requires an explicit user gesture + token per call; nothing is stored.");
  }
  const token = getAuthToken();
  const headers: Record<string, string> = {
    // Per-call Puter user token (never persisted, never logged server-side).
    "X-Puter-Token": auth.userToken,
  };
  if (token) {
    // App JWT stays on Authorization so the backend ownership check runs.
    headers["Authorization"] = `Bearer ${token}`;
  }
  const response = await fetch(
    `${API_BASE_URL}/api/ocr/puter/document/${encodeURIComponent(documentId)}`,
    { method: "POST", headers }
  );
  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: "Puter processing degraded" }));
    // Explicit degraded (never silent): surface the reason to the caller.
    throw new Error(error.detail || "Puter processing degraded explicitly");
  }
  return response.json();
};