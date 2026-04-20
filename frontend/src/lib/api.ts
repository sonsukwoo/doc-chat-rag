import type {
  ThreadChatTurnResult,
  ThreadChatViewResponse,
  ThreadDeleteResponse,
  DocumentInfoResponse,
  ThreadDocumentPipelineResponse,
  ThreadDocumentRecord,
  ThreadRecord,
  ReviewDecisionsPayload,
  ReviewSourceResponse,
} from "../types";

const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL?.replace(/\/$/, "") ||
  (import.meta.env.DEV ? "/api" : "http://127.0.0.1:8000");

function encodePathSegment(value: string): string {
  return encodeURIComponent(String(value ?? ""));
}

function isReadRequest(init?: RequestInit): boolean {
  return !init?.method || init.method.toUpperCase() === "GET";
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const maxAttempts = isReadRequest(init) ? 3 : 1;
  let lastError: unknown = null;

  for (let attempt = 1; attempt <= maxAttempts; attempt += 1) {
    try {
      const response = await fetch(`${API_BASE_URL}${path}`, init);
      if (!response.ok) {
        const text = await response.text();
        let message = text || `HTTP ${response.status}`;
        try {
          const parsed = JSON.parse(text) as { detail?: string };
          message = parsed.detail || message;
        } catch {
          // plain text error는 원문을 그대로 유지한다.
        }
        throw new Error(message);
      }
      return (await response.json()) as T;
    } catch (error) {
      lastError = error;
      if (attempt < maxAttempts) {
        await sleep(250 * attempt);
        continue;
      }
    }
  }

  const fallbackMessage =
    lastError instanceof Error && lastError.message
      ? lastError.message
      : "알 수 없는 네트워크 오류";
  throw new Error(`API 요청 실패 (${path}): ${fallbackMessage}`);
}

export function getApiBaseUrl(): string {
  return API_BASE_URL;
}

export function buildStageAssetUrl(
  documentId: string,
  stage: "source" | "stage1" | "stage2" | "review" | "stage3" | "stage4",
  relativePath: string,
): string {
  const normalized = relativePath.replace(/^\/+/, "");
  return `${API_BASE_URL}/documents/${documentId}/assets/${stage}/${normalized}`;
}

export async function listDocuments(): Promise<{ documents: DocumentInfoResponse["document"][] }> {
  return request("/documents");
}

export async function getDocument(
  documentId: string,
): Promise<DocumentInfoResponse> {
  return request(`/documents/${documentId}`);
}

export async function uploadDocument(file: File): Promise<DocumentInfoResponse> {
  const formData = new FormData();
  formData.append("file", file);
  return request("/documents/upload", {
    method: "POST",
    body: formData,
  });
}

export async function runStage(
  documentId: string,
  stage: "stage1" | "stage2" | "stage3",
): Promise<unknown> {
  return request(`/documents/${documentId}/${stage}`, {
    method: "POST",
  });
}

export async function getReviewSource(
  documentId: string,
): Promise<ReviewSourceResponse> {
  return request(`/documents/${documentId}/review/source`);
}

export async function saveReviewDecisions(
  documentId: string,
  payload: ReviewDecisionsPayload,
): Promise<unknown> {
  return request(`/documents/${documentId}/review/decisions`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });
}

export async function applyReview(documentId: string): Promise<unknown> {
  return request(`/documents/${documentId}/review/apply`, {
    method: "POST",
  });
}

export async function listThreads(
  includeArchived = false,
): Promise<{ threads: ThreadRecord[] }> {
  const query = includeArchived ? "?include_archived=true" : "";
  return request(`/threads${query}`);
}

export async function getThread(threadId: string): Promise<{ thread: ThreadRecord }> {
  return request(`/threads/${encodePathSegment(threadId)}`);
}

export async function deleteThread(
  threadId: string,
): Promise<ThreadDeleteResponse> {
  return request(`/threads/${encodePathSegment(threadId)}`, {
    method: "DELETE",
  });
}

export async function getThreadChat(
  threadId: string,
): Promise<ThreadChatViewResponse> {
  return request(`/threads/${encodePathSegment(threadId)}/chat`);
}

export async function getThreadDocuments(
  threadId: string,
): Promise<{ documents: ThreadDocumentRecord[] }> {
  return request(`/threads/${encodePathSegment(threadId)}/documents`);
}

export async function bootstrapThread(
  params: {
    threadName: string;
    file: File;
    description?: string;
    defaultRetrievalMode?: "dense" | "hybrid";
  },
): Promise<ThreadDocumentPipelineResponse> {
  const formData = new FormData();
  formData.append("thread_name", params.threadName);
  formData.append("file", params.file);
  if (params.description) {
    formData.append("description", params.description);
  }
  formData.append("default_retrieval_mode", params.defaultRetrievalMode || "dense");
  return request("/threads/bootstrap", {
    method: "POST",
    body: formData,
  });
}

export async function processThreadDocumentUpload(
  threadId: string,
  file: File,
): Promise<ThreadDocumentPipelineResponse> {
  const formData = new FormData();
  formData.append("file", file);
  return request(`/threads/${encodePathSegment(threadId)}/documents/process-upload`, {
    method: "POST",
    body: formData,
  });
}

export async function prepareThreadDocumentForReview(
  threadId: string,
  documentId: string,
): Promise<ThreadDocumentPipelineResponse> {
  return request(
    `/threads/${encodePathSegment(threadId)}/documents/${encodePathSegment(documentId)}/prepare-review`,
    {
      method: "POST",
    },
  );
}

export async function finalizeThreadDocumentReview(
  threadId: string,
  documentId: string,
): Promise<ThreadDocumentPipelineResponse> {
  return request(
    `/threads/${encodePathSegment(threadId)}/documents/${encodePathSegment(documentId)}/finalize-review`,
    {
      method: "POST",
    },
  );
}

export async function sendThreadChatMessage(
  threadId: string,
  payload: {
    message: string;
    resume?: boolean;
    allowWebSearch?: boolean;
  },
): Promise<{ result: ThreadChatTurnResult }> {
  return request(`/threads/${encodePathSegment(threadId)}/chat`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      message: payload.message,
      resume: payload.resume ?? false,
      allow_web_search: payload.allowWebSearch ?? false,
    }),
  });
}
