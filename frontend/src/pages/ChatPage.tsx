import { useEffect, useMemo, useRef, useState } from "react";
import type { DragEvent, KeyboardEvent } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";

import { ChatTraceDetails } from "../components/ChatTraceDetails";
import { ThreadSidebar } from "../components/ThreadSidebar";
import {
  buildStageAssetUrl,
  deleteThread,
  getThreadChat,
  getThreadDocuments,
  listThreads,
  processThreadDocumentUpload,
  sendThreadChatMessage,
} from "../lib/api";
import {
  isThreadReady,
} from "../lib/threadUi";
import type {
  ChatCitation,
  ChatDebugTrace,
  ChatEvidenceChunk,
  ChatVisualAsset,
  ThreadChatHistoryMessage,
  ThreadDocumentRecord,
  ThreadRecord,
} from "../types";

type ChatMessage = {
  id: string;
  role: "user" | "assistant";
  content: string;
  kind: "answer" | "interrupt" | "error" | "pending";
  createdAt?: string | null;
  citations?: ChatCitation[];
  evidenceChunks?: ChatEvidenceChunk[];
  visualAssets?: ChatVisualAsset[];
  retrievalMode?: string;
  debugTrace?: ChatDebugTrace | null;
};

const EMPTY_STATE_SUGGESTIONS = [
  "이 문서 핵심 주제를 3문장으로 요약해줘",
  "주요 표와 그림이 무엇을 말하는지 정리해줘",
  "가장 중요한 근거 문단만 뽑아서 설명해줘",
];

function canOpenReview(document: ThreadDocumentRecord): boolean {
  const stages = document.stages || {};
  return (
    stages.stage2?.status === "completed" ||
    stages.review?.status === "running" ||
    stages.review?.status === "completed" ||
    stages.stage3?.status === "completed"
  );
}

function getDocumentChatStatus(document: ThreadDocumentRecord): string {
  const stages = document.stages || {};
  if (stages.stage3?.status === "completed") {
    return "검색 가능";
  }
  if (stages.review?.status === "completed") {
    return "인덱싱 대기";
  }
  if (canOpenReview(document)) {
    return "검수 가능";
  }
  if (stages.stage2?.status === "running") {
    return "준비 중";
  }
  return "업로드됨";
}

function canChat(thread: ThreadRecord | null, documents: ThreadDocumentRecord[]): boolean {
  if (isThreadReady(thread)) {
    return true;
  }
  return documents.some((document) => document.stages?.stage3?.status === "completed");
}

function hydratePersistedMessages(records: ThreadChatHistoryMessage[]): ChatMessage[] {
  return records.map((record, index) => ({
    id: `persisted-${index}-${record.role}`,
    role: record.role,
    content: record.content,
    kind: record.kind,
    createdAt: record.created_at || null,
    citations: record.citations,
    visualAssets: record.visual_assets,
    evidenceChunks: record.evidence_chunks,
    retrievalMode: record.retrieval_mode || undefined,
    debugTrace: record.debug_trace,
  }));
}

function formatChatBubbleTime(value?: string | null): string | null {
  if (!value) {
    return null;
  }

  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return null;
  }

  return new Intl.DateTimeFormat("ko-KR", {
    hour: "numeric",
    minute: "2-digit",
    hour12: true,
  }).format(date);
}

function ThreadPanelIcon(props: { open: boolean }) {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true" className={`thread-panel-icon ${props.open ? "is-open" : ""}`}>
      <path
        d="M4 7h16M7 12h10M10 17h4"
        fill="none"
        stroke="currentColor"
        strokeLinecap="round"
        strokeWidth="2"
      />
    </svg>
  );
}

function SendArrowIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path
        d="M12 5v14m0-14 5 5m-5-5-5 5"
        fill="none"
        stroke="currentColor"
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth="2"
      />
    </svg>
  );
}

export function ChatPage() {
  const { threadId = "" } = useParams();
  const navigate = useNavigate();

  const [threads, setThreads] = useState<ThreadRecord[]>([]);
  const [thread, setThread] = useState<ThreadRecord | null>(null);
  const [documents, setDocuments] = useState<ThreadDocumentRecord[]>([]);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [inputValue, setInputValue] = useState("");
  const [loading, setLoading] = useState(true);
  const [sending, setSending] = useState(false);
  const [historyNotice, setHistoryNotice] = useState<string | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [showThreadPanel, setShowThreadPanel] = useState(false);
  const [appendFile, setAppendFile] = useState<File | null>(null);
  const [uploadingDocument, setUploadingDocument] = useState(false);
  const [deletingThreadId, setDeletingThreadId] = useState<string | null>(null);
  const [liveProgressIndex, setLiveProgressIndex] = useState(0);
  const transcriptRef = useRef<HTMLElement | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);
  const chatReady = canChat(thread, documents);

  const sortedDocuments = useMemo(
    () =>
      [...documents].sort((a, b) =>
        String(b.uploaded_at || "").localeCompare(String(a.uploaded_at || "")),
      ),
    [documents],
  );

  async function loadThreadChatView(targetThreadId: string) {
    const [threadsResponse, chatResponse, documentsResponse] = await Promise.all([
      listThreads(),
      getThreadChat(targetThreadId),
      getThreadDocuments(targetThreadId),
    ]);

    const nextMessages = hydratePersistedMessages(chatResponse.messages || []);

    setThreads(threadsResponse.threads);
    setThread(chatResponse.thread);
    setDocuments(documentsResponse.documents);
    setMessages(nextMessages);
    setHistoryNotice(chatResponse.history_notice || null);
  }

  useEffect(() => {
    if (!threadId) {
      navigate("/", { replace: true });
      return;
    }

    setLoading(true);
    setErrorMessage(null);
    setThread(null);
    setDocuments([]);
    setMessages([]);
    setHistoryNotice(null);
    setInputValue("");
    setAppendFile(null);

    let mounted = true;

    void loadThreadChatView(threadId)
      .catch((error) => {
        if (!mounted) {
          return;
        }
        setErrorMessage(
          error instanceof Error ? error.message : "채팅 스레드 정보를 불러오지 못했습니다.",
        );
      })
      .finally(() => {
        if (mounted) {
          setLoading(false);
        }
      });

    return () => {
      mounted = false;
    };
  }, [navigate, threadId]);

  useEffect(() => {
    if (!sending) {
      setLiveProgressIndex(0);
      return;
    }

    setLiveProgressIndex(0);
    const timer = window.setInterval(() => {
      setLiveProgressIndex((current) => Math.min(current + 1, 4));
    }, 1200);

    return () => {
      window.clearInterval(timer);
    };
  }, [sending]);

  useEffect(() => {
    const node = transcriptRef.current;
    if (!node) {
      return;
    }
    node.scrollTop = node.scrollHeight;
  }, [messages, loading]);

  function resizeComposer() {
    const node = textareaRef.current;
    if (!node) {
      return;
    }
    node.style.height = "0px";
    node.style.height = `${Math.min(node.scrollHeight, 220)}px`;
  }

  useEffect(() => {
    resizeComposer();
  }, [inputValue, chatReady]);

  function handleComposerKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key !== "Enter") {
      return;
    }
    if (event.shiftKey) {
      return;
    }
    event.preventDefault();
    void handleSendMessage();
  }

  async function handleSendMessage(options?: {
    message?: string;
  }) {
    const trimmed = String(options?.message ?? inputValue).trim();
    if (!trimmed || !threadId || sending || !chatReady) {
      return;
    }
    const activeThreadId = threadId;

    const userMessage: ChatMessage = {
      id: `${Date.now()}-user`,
      role: "user",
      content: trimmed,
      kind: "answer",
      createdAt: new Date().toISOString(),
    };
    const pendingMessageId = `${Date.now()}-assistant-pending`;
    const pendingMessage: ChatMessage = {
      id: pendingMessageId,
      role: "assistant",
      content: "질문을 해석하고 답변을 준비하고 있습니다.",
      kind: "pending",
      createdAt: new Date().toISOString(),
    };

    setMessages((current) => [...current, userMessage, pendingMessage]);
    if (!options?.message) {
      setInputValue("");
    }
    setSending(true);
    setErrorMessage(null);

    try {
      const response = await sendThreadChatMessage(activeThreadId, {
        message: trimmed,
        resume: false,
      });
      const result = response.result;

      try {
        await loadThreadChatView(activeThreadId);
      } catch {
        const assistantContent =
          result.status === "interrupted"
            ? [result.interrupt?.question, result.interrupt?.reason]
                .filter(Boolean)
                .join("\n\n")
            : result.final_answer || "답변을 생성하지 못했습니다.";
        const debugTrace =
          result.debug_trace ||
          (result.logs?.length || result.retrieval_mode
            ? {
                logs: result.logs,
                retrieval_mode: result.retrieval_mode || null,
              }
            : null);

        const assistantMessage: ChatMessage = {
          id: pendingMessageId,
          role: "assistant",
          content: assistantContent,
          kind: result.status === "interrupted" ? "interrupt" : "answer",
          createdAt: new Date().toISOString(),
          citations: result.citations,
          evidenceChunks: result.evidence_chunks,
          visualAssets: result.visual_assets,
          retrievalMode: result.retrieval_mode,
          debugTrace,
        };

        setMessages((current) =>
          current.map((message) => (message.id === pendingMessageId ? assistantMessage : message)),
        );
      }
    } catch (error) {
      const message =
        error instanceof Error ? error.message : "채팅 요청에 실패했습니다.";
      setErrorMessage(message);
      setMessages((current) => [
        ...current.map((item) =>
          item.id === pendingMessageId
            ? {
                id: pendingMessageId,
                role: "assistant" as const,
                content: message,
                kind: "error" as const,
                createdAt: new Date().toISOString(),
              }
            : item,
        ),
      ]);
    } finally {
      setSending(false);
    }
  }

  async function handleAppendDocument() {
    if (!threadId || !appendFile || uploadingDocument) {
      return;
    }

    setUploadingDocument(true);
    setErrorMessage(null);
    try {
      const result = await processThreadDocumentUpload(threadId, appendFile);
      setAppendFile(null);
      navigate(
        `/threads/${encodeURIComponent(result.thread.thread_id)}/documents/${encodeURIComponent(
          result.document.document_id,
        )}/review`,
      );
    } catch (error) {
      setErrorMessage(
        error instanceof Error ? error.message : "문서 추가와 준비에 실패했습니다.",
      );
    } finally {
      setUploadingDocument(false);
    }
  }

  async function handleDeleteThread(targetThreadId = threadId) {
    if (!targetThreadId || deletingThreadId) {
      return;
    }

    const confirmed = window.confirm(
      "이 채팅방을 삭제하면 연결 문서 메타데이터, 체크포인터, Qdrant 인덱스와 로컬 산출물이 함께 제거됩니다. 계속하시겠습니까?",
    );
    if (!confirmed) {
      return;
    }

    setDeletingThreadId(targetThreadId);
    setErrorMessage(null);
    try {
      const response = await deleteThread(targetThreadId);
      if (response.cleanup_warnings.length > 0) {
        window.alert(
          `채팅방 삭제는 완료됐지만 일부 외부 정리를 확인해야 합니다.\n\n${response.cleanup_warnings.join("\n")}`,
        );
      }
      if (targetThreadId === threadId) {
        navigate("/", { replace: true });
        return;
      }
      setThreads((current) => current.filter((thread) => thread.thread_id !== targetThreadId));
    } catch (error) {
      setErrorMessage(
        error instanceof Error ? error.message : "채팅방 삭제에 실패했습니다.",
      );
    } finally {
      setDeletingThreadId(null);
    }
  }

  function onAppendDrop(event: DragEvent<HTMLLabelElement>) {
    event.preventDefault();
    const file = event.dataTransfer.files?.[0];
    if (file) {
      setAppendFile(file);
    }
  }

  return (
    <div className="workspace-shell workspace-shell--chat app-shell">
      <ThreadSidebar
        threads={threads}
        selectedThreadId={threadId}
        onSelectThread={(nextThreadId) => navigate(`/threads/${encodeURIComponent(nextThreadId)}/chat`)}
        onCreateThread={() => navigate("/")}
        onDeleteThread={(targetThreadId) => void handleDeleteThread(targetThreadId)}
        deletingThreadId={deletingThreadId}
      />

      <main className="chat-main--refined">
        <header className="chat-header chat-header--minimal">
          <div className="chat-header-copy">
            <div className="chat-header-title-row">
              <h2>{thread?.thread_name || "채팅방"}</h2>
            </div>
          </div>

          <div className="chat-header-actions">
            <button
              className={`icon-button ${showThreadPanel ? "is-active" : ""}`}
              type="button"
              aria-label="채팅방 정보 열기"
              onClick={() => setShowThreadPanel((current) => !current)}
            >
              <ThreadPanelIcon open={showThreadPanel} />
            </button>
          </div>
        </header>

        {errorMessage ? <div className="error-banner">{errorMessage}</div> : null}
        {!loading && historyNotice && messages.length === 0 ? (
          <div className="notice-banner">{historyNotice}</div>
        ) : null}

        <section className={`chat-body-grid ${showThreadPanel ? "has-panel" : ""}`}>
          <section ref={transcriptRef} className="chat-transcript">
            {loading ? (
              <div className="empty-state">채팅 스레드를 불러오는 중입니다.</div>
            ) : messages.length === 0 ? (
              <div className="chat-empty-state">
                <p className="eyebrow">{chatReady ? "Ask" : "Ready"}</p>
                <h3>
                  {chatReady
                    ? `${thread?.thread_name || "문서"} 대화를 시작하세요`
                    : "검수가 끝나면 이 자리에서 대화를 이어갑니다"}
                </h3>
                <p className="muted-text">
                  {chatReady
                    ? historyNotice
                      ? `${historyNotice} 첫 질문을 보내면 이 자리부터 대화가 누적됩니다.`
                      : "이 스레드의 이전 대화는 여기 누적되고, 질문은 현재 연결 문서 범위에서만 답합니다."
                    : "문서 준비 상태와 추가 업로드는 우측 상단 아이콘 패널에서 볼 수 있습니다."}
                </p>
                {chatReady ? (
                  <div className="chat-suggestion-list">
                    {EMPTY_STATE_SUGGESTIONS.map((suggestion) => (
                      <button
                        key={suggestion}
                        className="chat-suggestion-chip"
                        onClick={() => setInputValue(suggestion)}
                      >
                        {suggestion}
                      </button>
                    ))}
                  </div>
                ) : null}
              </div>
            ) : (
              messages.map((message) => {
                const formattedTime = formatChatBubbleTime(message.createdAt);

                return (
                <article
                  key={message.id}
                  className={`chat-bubble chat-bubble-${message.role} ${
                    message.kind === "interrupt" ? "is-interrupt" : ""
                  } ${message.kind === "error" ? "is-error" : ""}`}
                >
                  <div className="chat-bubble-meta">
                    <strong>{message.role === "user" ? "나" : "Doc Chat"}</strong>
                  </div>
                  <div className="chat-bubble-content">
                    {message.content.split("\n").map((line, index) => (
                      <p key={`${message.id}-${index}`}>{line || "\u00a0"}</p>
                    ))}
                  </div>
                  {formattedTime ? (
                    <div className="chat-bubble-footer">
                      <span className="chat-bubble-time">{formattedTime}</span>
                    </div>
                  ) : null}

                  {message.kind === "pending" ? (
                    <ChatTraceDetails
                      pending
                      liveProgressIndex={liveProgressIndex}
                      documents={documents}
                    />
                  ) : null}

                  {message.visualAssets && message.visualAssets.length > 0 ? (
                    <div className="chat-visual-grid">
                      {message.visualAssets.map((asset) => (
                        <article key={asset.asset_ref} className="chat-visual-card">
                          <img
                            src={buildStageAssetUrl(
                              asset.document_id,
                              asset.asset_stage,
                              asset.relative_path,
                            )}
                            alt={asset.caption || asset.summary_text || asset.asset_ref}
                          />
                          <div className="chat-visual-copy">
                            <strong>{asset.caption || asset.summary_text || asset.asset_kind}</strong>
                            <span>
                              {asset.page ? `p.${asset.page}` : asset.pages?.join(", ") || ""}
                            </span>
                          </div>
                        </article>
                      ))}
                    </div>
                  ) : null}

                  {message.kind !== "pending" &&
                  ((message.citations && message.citations.length > 0) || message.debugTrace) ? (
                    <div className="chat-citation-summary">
                      {message.citations && message.citations.length > 0 ? (
                        <span className="detail-chip">근거 {message.citations.length}개</span>
                      ) : null}
                      {message.retrievalMode ? (
                        <span className="detail-chip detail-chip-muted">
                          실행 {message.retrievalMode}
                        </span>
                      ) : null}
                    </div>
                  ) : null}

                  {message.kind !== "pending" ? (
                    <ChatTraceDetails
                      debugTrace={message.debugTrace}
                      evidenceChunks={message.evidenceChunks}
                      documents={documents}
                      retrievalMode={message.retrievalMode}
                    />
                  ) : null}
                </article>
                );
              })
            )}
          </section>

          {showThreadPanel ? (
            <aside className="chat-sidepanel">
              <section className="chat-sidepanel-section">
                <div className="section-card-header">
                  <div>
                    <p className="eyebrow">Overview</p>
                    <h3>현재 상태</h3>
                  </div>
                </div>
                <div className="thread-overview-grid thread-overview-grid--panel">
                  <div className="review-stat-card">
                    <strong>{thread?.document_count || 0}</strong>
                    <span>연결 문서</span>
                  </div>
                  <div className="review-stat-card">
                    <strong>{thread?.default_retrieval_mode || "dense"}</strong>
                    <span>저장된 스레드 설정</span>
                  </div>
                </div>
              </section>

              <section className="chat-sidepanel-section">
                <div className="section-card-header">
                  <div>
                    <p className="eyebrow">Add</p>
                    <h3>문서 추가</h3>
                  </div>
                </div>

                <label
                  className={`upload-dropzone compact chat-panel-dropzone ${
                    uploadingDocument ? "is-busy" : ""
                  }`}
                  onDragOver={(event) => event.preventDefault()}
                  onDrop={onAppendDrop}
                >
                  <input
                    type="file"
                    accept="application/pdf"
                    onChange={(event) => setAppendFile(event.target.files?.[0] || null)}
                  />
                  <div className="upload-dropzone-content">
                    <strong>{appendFile ? appendFile.name : "추가할 PDF를 선택하세요"}</strong>
                    <span>준비가 끝나면 바로 검수 화면으로 이동합니다.</span>
                  </div>
                </label>

                <div className="detail-actions">
                  <button
                    className="secondary-button"
                    onClick={() => void handleAppendDocument()}
                    disabled={!appendFile || uploadingDocument}
                  >
                    {uploadingDocument ? "문서 준비 중..." : "문서 추가"}
                  </button>
                </div>
              </section>

              <section className="chat-sidepanel-section">
                <div className="section-card-header">
                  <div>
                    <p className="eyebrow">Documents</p>
                    <h3>연결 문서</h3>
                  </div>
                </div>

                <div className="chat-panel-document-list">
                  {sortedDocuments.length === 0 ? (
                    <div className="empty-state">연결된 문서가 없습니다.</div>
                  ) : null}

                  {sortedDocuments.map((document) => (
                    <article key={document.document_id} className="chat-panel-document-item">
                      <div className="chat-panel-document-top">
                        <strong>{document.original_filename}</strong>
                        <span>{getDocumentChatStatus(document)}</span>
                      </div>
                      <div className="chat-panel-document-actions">
                        <Link
                          className="ghost-link"
                          to={`/threads/${encodeURIComponent(threadId)}/documents/${encodeURIComponent(
                            document.document_id,
                          )}/review`}
                        >
                          검수 열기
                        </Link>
                      </div>
                    </article>
                  ))}
                </div>
              </section>

              <section className="chat-sidepanel-section">
                <div className="section-card-header">
                  <div>
                    <p className="eyebrow">Manage</p>
                    <h3>채팅방 삭제</h3>
                  </div>
                </div>
                <p className="muted-text">
                  thread 메타데이터, 연결 문서 DB, 체크포인터, Qdrant 인덱스를 함께 제거합니다.
                </p>
                <div className="detail-actions">
                  <button
                    className="danger-button"
                    type="button"
                    onClick={() => void handleDeleteThread()}
                    disabled={Boolean(deletingThreadId)}
                  >
                    {deletingThreadId === threadId ? "삭제 중..." : "채팅방 삭제"}
                  </button>
                </div>
              </section>
            </aside>
          ) : null}
        </section>

        <footer className="chat-composer--refined">
          <div className="chat-composer-shell">
            <textarea
              ref={textareaRef}
              className="chat-input"
              value={inputValue}
              onChange={(event) => setInputValue(event.target.value)}
              onKeyDown={handleComposerKeyDown}
              placeholder={
                !chatReady
                  ? "검수와 stage3가 끝나면 질문할 수 있습니다."
                  : "문서에 대해 질문하세요."
              }
              disabled={sending || !chatReady}
              rows={1}
            />
            <div className="chat-composer-actions">
              <span className="muted-text chat-composer-hint">
                Enter 전송 · Shift+Enter 줄바꿈
              </span>
              <button
                className="chat-send-button"
                onClick={() => void handleSendMessage()}
                disabled={sending || !inputValue.trim() || !chatReady}
                aria-label={sending ? "전송 중" : "질문 보내기"}
              >
                <SendArrowIcon />
              </button>
            </div>
          </div>
        </footer>
      </main>
    </div>
  );
}
