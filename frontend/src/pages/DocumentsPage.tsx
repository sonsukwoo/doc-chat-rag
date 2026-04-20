import { useEffect, useMemo, useState } from "react";
import type { DragEvent, FormEvent } from "react";
import { useNavigate } from "react-router-dom";

import { ThreadSidebar } from "../components/ThreadSidebar";
import { bootstrapThread, deleteThread, listThreads } from "../lib/api";
import type { ThreadRecord } from "../types";

export function DocumentsPage() {
  const navigate = useNavigate();

  const [threads, setThreads] = useState<ThreadRecord[]>([]);
  const [threadName, setThreadName] = useState("");
  const [defaultRetrievalMode, setDefaultRetrievalMode] = useState<"dense" | "hybrid">(
    "dense",
  );
  const [bootstrapFile, setBootstrapFile] = useState<File | null>(null);
  const [busy, setBusy] = useState(false);
  const [deletingThreadId, setDeletingThreadId] = useState<string | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const sortedThreads = useMemo(
    () =>
      [...threads].sort((a, b) =>
        String(b.updated_at || "").localeCompare(String(a.updated_at || "")),
      ),
    [threads],
  );

  async function refreshThreads() {
    const response = await listThreads();
    setThreads(response.threads);
  }

  useEffect(() => {
    void refreshThreads().catch((error) => {
      setErrorMessage(
        error instanceof Error ? error.message : "thread 목록을 불러오지 못했습니다.",
      );
    });
  }, []);

  async function handleBootstrapSubmit(event?: FormEvent<HTMLFormElement>) {
    event?.preventDefault();

    if (!bootstrapFile) {
      setErrorMessage("첫 PDF 문서를 선택해야 합니다.");
      return;
    }
    if (!threadName.trim()) {
      setErrorMessage("채팅방 이름을 입력해야 합니다.");
      return;
    }

    setBusy(true);
    setErrorMessage(null);
    try {
      const result = await bootstrapThread({
        threadName: threadName.trim(),
        defaultRetrievalMode,
        file: bootstrapFile,
      });
      setThreadName("");
      setBootstrapFile(null);
      await refreshThreads();
      navigate(
        `/threads/${encodeURIComponent(result.thread.thread_id)}/documents/${encodeURIComponent(
          result.document.document_id,
        )}/review`,
      );
    } catch (error) {
      setErrorMessage(
        error instanceof Error ? error.message : "채팅방 생성과 문서 준비에 실패했습니다.",
      );
    } finally {
      setBusy(false);
    }
  }

  function onBootstrapDrop(event: DragEvent<HTMLLabelElement>) {
    event.preventDefault();
    const file = event.dataTransfer.files?.[0];
    if (file) {
      setBootstrapFile(file);
    }
  }

  async function handleDeleteThread(threadId: string) {
    if (!threadId || deletingThreadId) {
      return;
    }

    const confirmed = window.confirm(
      "이 채팅방을 삭제하면 연결 문서 메타데이터, 체크포인터, Qdrant 인덱스와 로컬 산출물이 함께 제거됩니다. 계속하시겠습니까?",
    );
    if (!confirmed) {
      return;
    }

    setDeletingThreadId(threadId);
    setErrorMessage(null);
    try {
      const response = await deleteThread(threadId);
      if (response.cleanup_warnings.length > 0) {
        window.alert(
          `채팅방 삭제는 완료됐지만 일부 외부 정리를 확인해야 합니다.\n\n${response.cleanup_warnings.join("\n")}`,
        );
      }
      setThreads((current) => current.filter((thread) => thread.thread_id !== threadId));
    } catch (error) {
      setErrorMessage(
        error instanceof Error ? error.message : "채팅방 삭제에 실패했습니다.",
      );
    } finally {
      setDeletingThreadId(null);
    }
  }

  return (
    <div className="workspace-shell workspace-shell--chat app-shell">
      <ThreadSidebar
        threads={sortedThreads}
        selectedThreadId={null}
        onSelectThread={(threadId) => navigate(`/threads/${encodeURIComponent(threadId)}/chat`)}
        onCreateThread={() => {
          setErrorMessage(null);
          setThreadName("");
          setBootstrapFile(null);
        }}
        onDeleteThread={(threadId) => void handleDeleteThread(threadId)}
        deletingThreadId={deletingThreadId}
      />

      <main className="workspace-main workspace-main--chat thread-home-main">
        {errorMessage ? <div className="error-banner">{errorMessage}</div> : null}

        <section className="thread-launcher">
          <div className="thread-launcher-copy">
            <p className="eyebrow">Start</p>
            <h2>
              {sortedThreads.length === 0
                ? "첫 채팅방을 만들고 문서 검수부터 시작하세요"
                : "새 채팅방을 만들고 PDF를 바로 올리세요"}
            </h2>
            <p className="muted-text">
              문서를 올리면 stage1, stage2 후 바로 검수 화면으로 이동하고, 끝나면 채팅방별로
              대화를 이어갈 수 있습니다.
            </p>
          </div>

          <form
            className="thread-launcher-card"
            onSubmit={(event) => void handleBootstrapSubmit(event)}
          >
            <div className="thread-launcher-row">
              <label className="field-group">
                <span>채팅방 이름</span>
                <input
                  className="input"
                  value={threadName}
                  onChange={(event) => setThreadName(event.target.value)}
                  placeholder="예: 피부질환 논문 QA"
                />
              </label>

              <label className="field-group">
                <span>기본 검색 모드</span>
                <select
                  className="input"
                  value={defaultRetrievalMode}
                  onChange={(event) =>
                    setDefaultRetrievalMode(event.target.value as "dense" | "hybrid")
                  }
                >
                  <option value="dense">dense</option>
                  <option value="hybrid">hybrid</option>
                </select>
              </label>
            </div>

            <div className="thread-launcher-steps">
              <span>1. PDF 업로드</span>
              <span>2. 검수 진행</span>
              <span>3. 채팅 시작</span>
            </div>

            <label
              className={`upload-dropzone compact thread-launcher-dropzone ${busy ? "is-busy" : ""}`}
              onDragOver={(event) => event.preventDefault()}
              onDrop={onBootstrapDrop}
            >
              <input
                type="file"
                accept="application/pdf"
                onChange={(event) => setBootstrapFile(event.target.files?.[0] || null)}
              />
              <div className="upload-dropzone-content">
                <strong>{bootstrapFile ? bootstrapFile.name : "첫 PDF 문서를 선택하세요"}</strong>
                <span>업로드 직후 검수 화면으로 이동합니다.</span>
              </div>
            </label>

            <div className="detail-actions">
              <button className="primary-button thread-primary-action" type="submit" disabled={busy}>
                {busy ? "준비 중..." : "채팅방 만들고 검수 시작"}
              </button>
            </div>
          </form>
        </section>
      </main>
    </div>
  );
}
