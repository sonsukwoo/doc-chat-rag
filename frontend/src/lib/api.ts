import type {
  DocumentInfoResponse,
  ReviewDecisionsPayload,
  ReviewSourceResponse,
} from "../types";

const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL?.replace(/\/$/, "") ||
  "http://127.0.0.1:8000";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, init);
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `HTTP ${response.status}`);
  }
  return (await response.json()) as T;
}

export function getApiBaseUrl(): string {
  return API_BASE_URL;
}

export function buildStageAssetUrl(
  documentId: string,
  stage: "stage2" | "review",
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
