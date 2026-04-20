import { useEffect, useMemo, useState } from "react";
import type { DragEvent } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";

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
  getThreadLifecycleLabel,
  getThreadLifecycleTone,
  isThreadReady,
} from "../lib/threadUi";
import type {
  ChatCitation,
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
  kind: "answer" | "interrupt" | "error";
  createdAt?: string | null;
  citations?: ChatCitation[];
  evidenceChunks?: ChatEvidenceChunk[];
  visualAssets?: ChatVisualAsset[];
  retrievalMode?: string;
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

function getDocumentLabel(
  documentId: string | undefined,
  documents: ThreadDocumentRecord[],
): string {
  const resolvedId = String(documentId || "").trim();
  const matched = documents.find((document) => document.document_id === resolvedId);
  return matched?.original_filename || resolvedId || "문서";
}

function hasPendingInterrupt(messages: ChatMessage[]): boolean {
  const lastMessage = messages[messages.length - 1];
  return lastMessage?.role === "assistant" && lastMessage.kind === "interrupt";
}

function hydratePersistedMessages(records: ThreadChatHistoryMessage[]): ChatMessage[] {
  return records.map((record, index) => ({
    id: `persisted-${index}-${record.role}`,
    role: record.role,
    content: record.content,
    kind: record.kind,
    createdAt: null,
  }));
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
  const [pendingInterrupt, setPendingInterrupt] = useState(false);
  const [historyNotice, setHistoryNotice] = useState<string | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [showThreadPanel, setShowThreadPanel] = useState(false);
  const [appendFile, setAppendFile] = useState<File | null>(null);
  const [uploadingDocument, setUploadingDocument] = useState(false);
  const [deletingThreadId, setDeletingThreadId] = useState<string | null>(null);

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
    setPendingInterrupt(hasPendingInterrupt(nextMessages));
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
    setPendingInterrupt(false);
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

  const chatReady = canChat(thread, documents);

  async function handleSendMessage() {
    const trimmed = inputValue.trim();
    if (!trimmed || !threadId || sending || !chatReady) {
      return;
    }

    const userMessage: ChatMessage = {
      id: `${Date.now()}-user`,
      role: "user",
      content: trimmed,
      kind: "answer",
      createdAt: new Date().toISOString(),
    };

    setMessages((current) => [...current, userMessage]);
    setInputValue("");
    setSending(true);
    setErrorMessage(null);

    try {
      const response = await sendThreadChatMessage(threadId, {
        message: trimmed,
        resume: pendingInterrupt,
      });
      const result = response.result;

      const assistantContent =
        result.status === "interrupted"
          ? [result.interrupt?.question, result.interrupt?.reason]
              .filter(Boolean)
              .join("\n\n")
          : result.final_answer || "답변을 생성하지 못했습니다.";

      const assistantMessage: ChatMessage = {
        id: `${Date.now()}-assistant`,
        role: "assistant",
        content: assistantContent,
        kind: result.status === "interrupted" ? "interrupt" : "answer",
        createdAt: new Date().toISOString(),
        citations: result.citations,
        evidenceChunks: result.evidence_chunks,
        visualAssets: result.visual_assets,
        retrievalMode: result.retrieval_mode,
      };

      setMessages((current) => [...current, assistantMessage]);
      setPendingInterrupt(result.status === "interrupted");
    } catch (error) {
      const message =
        error instanceof Error ? error.message : "채팅 요청에 실패했습니다.";
      setErrorMessage(message);
      setMessages((current) => [
        ...current,
        {
          id: `${Date.now()}-error`,
          role: "assistant",
          content: message,
          kind: "error",
          createdAt: new Date().toISOString(),
        },
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
        onRefresh={() => void loadThreadChatView(threadId).catch(() => undefined)}
        onDeleteThread={(targetThreadId) => void handleDeleteThread(targetThreadId)}
        deletingThreadId={deletingThreadId}
      />

      <main className="chat-main chat-main--refined">
        <header className="chat-header chat-header--minimal">
          <div className="chat-header-copy">
            <p className="eyebrow">Thread</p>
            <div className="chat-header-title-row">
              <h2>{thread?.thread_name || "채팅방"}</h2>
              <span className={`room-status-chip room-status-${getThreadLifecycleTone(thread)}`}>
                {getThreadLifecycleLabel(thread)}
              </span>
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
          <section className="chat-transcript">
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
              messages.map((message) => (
                <article
                  key={message.id}
                  className={`chat-bubble chat-bubble-${message.role} ${
                    message.kind === "interrupt" ? "is-interrupt" : ""
                  } ${message.kind === "error" ? "is-error" : ""}`}
                >
                  <div className="chat-bubble-meta">
                    <strong>{message.role === "user" ? "나" : "Doc Chat"}</strong>
                    {message.createdAt ? (
                      <span>{new Date(message.createdAt).toLocaleTimeString()}</span>
                    ) : null}
                  </div>
                  <div className="chat-bubble-content">
                    {message.content.split("\n").map((line, index) => (
                      <p key={`${message.id}-${index}`}>{line || "\u00a0"}</p>
                    ))}
                  </div>

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

                  {message.citations && message.citations.length > 0 ? (
                    <div className="chat-citation-summary">
                      <span className="detail-chip">근거 {message.citations.length}개</span>
                      {message.retrievalMode ? (
                        <span className="detail-chip detail-chip-muted">
                          {message.retrievalMode}
                        </span>
                      ) : null}
                    </div>
                  ) : null}

                  {message.evidenceChunks && message.evidenceChunks.length > 0 ? (
                    <details className="chat-evidence-details">
                      <summary className="chat-evidence-summary">
                        <div>
                          <strong>텍스트 근거</strong>
                          <span>{message.evidenceChunks.length}개 청크</span>
                        </div>
                      </summary>
                      <div className="chat-evidence-grid">
                        {message.evidenceChunks.map((chunk, index) => (
                          <article
                            key={`${message.id}-evidence-${chunk.chunk_id}-${index}`}
                            className="chat-evidence-card"
                          >
                            <div className="chat-evidence-meta">
                              <strong>{getDocumentLabel(chunk.document_id, documents)}</strong>
                              <span>
                                {chunk.page ? `p.${chunk.page}` : "page ?"}
                                {chunk.section_title ? ` · ${chunk.section_title}` : ""}
                              </span>
                            </div>
                            <p>{chunk.text_excerpt}</p>
                          </article>
                        ))}
                      </div>
                    </details>
                  ) : null}
                </article>
              ))
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
                    <span>검색 모드</span>
                  </div>
                  <div className="review-stat-card">
                    <strong>{getThreadLifecycleLabel(thread)}</strong>
                    <span>채팅 상태</span>
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

        <footer className="chat-composer chat-composer--refined">
          <textarea
            className="chat-input"
            value={inputValue}
            onChange={(event) => setInputValue(event.target.value)}
            placeholder={
              !chatReady
                ? "검수와 stage3가 끝나면 이 자리에서 질문할 수 있습니다."
                : pendingInterrupt
                  ? "추가 정보를 입력하면 같은 질문 흐름으로 재개됩니다."
                  : "문서에 대해 질문하세요."
            }
            disabled={sending || !chatReady}
            rows={4}
          />
          <div className="chat-composer-actions">
            <span className="muted-text">
              {!chatReady
                ? "채팅 준비 필요"
                : pendingInterrupt
                  ? "추가 정보 응답 모드"
                  : "현재 스레드 문서 범위 검색"}
            </span>
            <button
              className="primary-button"
              onClick={() => void handleSendMessage()}
              disabled={sending || !inputValue.trim() || !chatReady}
            >
              {sending ? "전송 중..." : pendingInterrupt ? "응답 보내기" : "질문하기"}
            </button>
          </div>
        </footer>
      </main>
    </div>
  );
}
