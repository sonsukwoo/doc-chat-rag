import type { ThreadRecord } from "../types";

export type ThreadLifecycleTone = "ready" | "review-pending" | "draft";

export function getThreadLifecycleLabel(
  thread: ThreadRecord | null | undefined,
): string {
  const lifecycle = String(thread?.metadata?.lifecycle_status || "draft");
  if (lifecycle === "ready") {
    return "채팅 가능";
  }
  if (lifecycle === "review_pending") {
    return "검수 필요";
  }
  return "준비 중";
}

export function getThreadLifecycleTone(
  thread: ThreadRecord | null | undefined,
): ThreadLifecycleTone {
  const lifecycle = String(thread?.metadata?.lifecycle_status || "draft");
  if (lifecycle === "ready") {
    return "ready";
  }
  if (lifecycle === "review_pending") {
    return "review-pending";
  }
  return "draft";
}

export function isThreadReady(
  thread: ThreadRecord | null | undefined,
): boolean {
  return String(thread?.metadata?.lifecycle_status || "") === "ready";
}
