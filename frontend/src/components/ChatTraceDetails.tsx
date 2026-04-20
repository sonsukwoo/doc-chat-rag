import type {
  ChatDebugTrace,
  ChatEvidenceChunk,
  ChatToolTrace,
  ThreadDocumentRecord,
} from "../types";

const LIVE_PROGRESS_STEPS = [
  "질문 해석",
  "문서 선택",
  "문서 검색",
  "근거 정리",
  "답변 작성",
];

function getDocumentLabel(
  documentId: string | undefined,
  documents: ThreadDocumentRecord[],
): string {
  const resolvedId = String(documentId || "").trim();
  const matched = documents.find((document) => document.document_id === resolvedId);
  return matched?.original_filename || resolvedId || "문서";
}

function describeLogEntry(entry: string): string {
  const normalized = String(entry || "").trim();
  if (!normalized) {
    return "";
  }
  if (normalized === "load_request_context") {
    return "요청 컨텍스트를 불러왔습니다.";
  }
  if (normalized.startsWith("classify_query:")) {
    const [, queryKind = "", selectionSource = ""] = normalized.split(":");
    return `질문을 ${queryKind} 유형으로 분류했습니다${selectionSource ? ` · ${selectionSource}` : ""}.`;
  }
  if (normalized.startsWith("respond_without_documents:")) {
    return "문서 검색 없이 직접 답변 경로로 처리했습니다.";
  }
  if (normalized === "agent_llm:tool_call") {
    return "모델이 도구 사용이 필요하다고 판단했습니다.";
  }
  if (normalized === "agent_llm:answer") {
    return "모델이 바로 답변 초안을 작성했습니다.";
  }
  if (normalized.startsWith("grounding_check:")) {
    const [, enough = "", source = ""] = normalized.split(":");
    return `근거 충족 여부를 확인했습니다 · ${enough}${source ? ` · ${source}` : ""}.`;
  }
  if (normalized.startsWith("fallback_or_retrieve_deeper:retrieved:")) {
    const [, , mode = "", count = ""] = normalized.split(":");
    return `추가 검색을 수행했습니다 · ${mode}${count ? ` · ${count}개` : ""}.`;
  }
  if (normalized.startsWith("fallback_or_retrieve_deeper:empty:")) {
    return "추가 검색을 수행했지만 근거를 찾지 못했습니다.";
  }
  if (normalized === "compose_answer_with_citations") {
    return "최종 답변과 근거를 정리했습니다.";
  }
  if (normalized.startsWith("clarify_if_needed:")) {
    return "질문 대상을 다시 확인하는 분기로 이동했습니다.";
  }
  return normalized;
}

function renderToolSummary(
  trace: ChatToolTrace,
  documents: ThreadDocumentRecord[],
): string | null {
  if (trace.name === "search_thread_knowledge") {
    const documentLabels = (trace.document_ids || [])
      .map((documentId) => getDocumentLabel(documentId, documents))
      .join(", ");
    const parts = [
      trace.query ? `질의: ${trace.query}` : "",
      documentLabels ? `대상 문서: ${documentLabels}` : "",
      trace.retrieval_mode ? `실행 검색: ${trace.retrieval_mode}` : "",
      typeof trace.top_k === "number" ? `top_k: ${trace.top_k}` : "",
      typeof trace.fetch_k === "number" ? `fetch_k: ${trace.fetch_k}` : "",
      typeof trace.retrieved_count === "number" ? `검색 청크: ${trace.retrieved_count}개` : "",
      trace.message || "",
    ];
    return parts.filter(Boolean).join(" · ");
  }

  if (trace.name === "expand_context_window") {
    const chunkCount = trace.chunk_ids?.length || 0;
    return [
      chunkCount > 0 ? `입력 청크: ${chunkCount}개` : "",
      typeof trace.block_count === "number" ? `확장 블록: ${trace.block_count}개` : "",
    ]
      .filter(Boolean)
      .join(" · ");
  }

  if (trace.name === "load_visual_asset") {
    return trace.asset_ref ? `asset_ref: ${trace.asset_ref}` : trace.message || null;
  }

  if (trace.name === "list_thread_documents") {
    const documentLabels = (trace.document_ids || [])
      .map((documentId) => getDocumentLabel(documentId, documents))
      .join(", ");
    return documentLabels || null;
  }

  if (trace.name === "web_search") {
    return [trace.query ? `질의: ${trace.query}` : "", trace.message || ""]
      .filter(Boolean)
      .join(" · ");
  }

  return trace.message || null;
}

function renderSearchTraceChips(trace?: ChatToolTrace | null): string[] {
  if (!trace || trace.name !== "search_thread_knowledge") {
    return [];
  }

  const chips: string[] = [];
  if (trace.per_document_search_used) {
    chips.push("문서별 병렬 검색");
  }
  if (trace.rerank_requested) {
    chips.push(trace.rerank_applied ? "리랭크 적용" : "리랭크 요청");
  } else if (trace.rerank_requested === false) {
    chips.push("리랭크 off");
  }
  if (trace.mmr_requested) {
    chips.push(trace.mmr_applied ? "MMR 적용" : "MMR 요청");
  } else if (trace.mmr_requested === false) {
    chips.push("MMR off");
  }
  if (trace.score_fallback_applied) {
    chips.push("threshold fallback");
  }
  return chips;
}

type ChatTraceDetailsProps = {
  debugTrace?: ChatDebugTrace | null;
  evidenceChunks?: ChatEvidenceChunk[];
  documents: ThreadDocumentRecord[];
  retrievalMode?: string;
  pending?: boolean;
  liveProgressIndex?: number;
};

export function ChatTraceDetails({
  debugTrace,
  evidenceChunks = [],
  documents,
  retrievalMode,
  pending = false,
  liveProgressIndex = 0,
}: ChatTraceDetailsProps) {
  if (pending) {
    return (
      <div className="chat-live-progress">
        <div className="chat-live-progress-head">
          <strong>답변 생성 중</strong>
          <span>{LIVE_PROGRESS_STEPS[Math.min(liveProgressIndex, LIVE_PROGRESS_STEPS.length - 1)]}</span>
        </div>
        <div className="chat-live-step-list">
          {LIVE_PROGRESS_STEPS.map((step, index) => (
            <span
              key={step}
              className={`chat-live-step ${
                index < liveProgressIndex
                  ? "is-complete"
                  : index === liveProgressIndex
                    ? "is-active"
                    : ""
              }`}
            >
              {step}
            </span>
          ))}
        </div>
      </div>
    );
  }

  const selectedDocumentIds = debugTrace?.selected_document_ids || [];
  const selectedDocumentQueries = debugTrace?.selected_document_queries || {};
  const toolCalls = debugTrace?.tool_calls || [];
  const searchTrace =
    toolCalls.find((trace) => trace.name === "search_thread_knowledge") || null;
  const logEntries = (debugTrace?.logs || []).map(describeLogEntry).filter(Boolean);
  const resolvedRetrievalMode =
    searchTrace?.retrieval_mode ||
    debugTrace?.executed_retrieval_mode ||
    retrievalMode ||
    debugTrace?.retrieval_mode ||
    null;
  const defaultRetrievalMode = debugTrace?.thread_default_retrieval_mode || null;
  const searchTraceChips = renderSearchTraceChips(searchTrace);
  const showImplicitRetrievalNotice =
    toolCalls.length === 0 && (evidenceChunks.length > 0 || Boolean(resolvedRetrievalMode));

  if (
    selectedDocumentIds.length === 0 &&
    toolCalls.length === 0 &&
    logEntries.length === 0 &&
    evidenceChunks.length === 0 &&
    !debugTrace?.model
  ) {
    return null;
  }

  return (
    <details className="chat-response-details">
      <summary className="chat-response-summary">
        <div className="chat-response-summary-head">
          <div className="chat-response-summary-copy">
            <strong>근거 및 진행 보기</strong>
            <span>
              {evidenceChunks.length}개 청크 · {toolCalls.length}개 툴 · {selectedDocumentIds.length}개 문서
            </span>
          </div>
          <span className="chat-response-summary-toggle" aria-hidden="true" />
        </div>
        <div className="chat-response-summary-chips">
          {debugTrace?.model ? <span className="detail-chip detail-chip-muted">{debugTrace.model}</span> : null}
          {resolvedRetrievalMode ? (
            <span className="detail-chip detail-chip-muted">{`실행 ${resolvedRetrievalMode}`}</span>
          ) : null}
          {searchTrace?.rerank_applied ? (
            <span className="detail-chip detail-chip-muted">리랭크 적용</span>
          ) : null}
        </div>
      </summary>

      <div className="chat-response-body">
        {defaultRetrievalMode || resolvedRetrievalMode || searchTrace ? (
          <section className="chat-response-section">
            <h4>실행 방식</h4>
            <div className="chat-trace-list">
              <article className="chat-trace-card">
                <strong>이번 답변 검색 경로</strong>
                <div className="detail-chip-row">
                  {defaultRetrievalMode ? (
                    <span className="detail-chip">기본 {defaultRetrievalMode}</span>
                  ) : null}
                  {resolvedRetrievalMode ? (
                    <span className="detail-chip detail-chip-muted">
                      실행 {resolvedRetrievalMode}
                    </span>
                  ) : null}
                  {searchTraceChips.map((chip) => (
                    <span key={chip} className="detail-chip">
                      {chip}
                    </span>
                  ))}
                </div>
                <p>
                  스레드 설정값과 별도로, 이번 답변에서 실제 실행된 검색 방식과 후처리 단계를
                  분리해서 보여줍니다.
                </p>
              </article>
            </div>
          </section>
        ) : null}

        {selectedDocumentIds.length > 0 ? (
          <section className="chat-response-section">
            <h4>선택 문서</h4>
            <div className="chat-trace-list">
              {selectedDocumentIds.map((documentId) => (
                <article key={documentId} className="chat-trace-card">
                  <strong>{getDocumentLabel(documentId, documents)}</strong>
                  <span>{documentId}</span>
                  {selectedDocumentQueries[documentId] ? (
                    <p>검색 질의: {selectedDocumentQueries[documentId]}</p>
                  ) : null}
                </article>
              ))}
            </div>
            {debugTrace?.selection_reason ? (
              <p className="chat-trace-note">{debugTrace.selection_reason}</p>
            ) : null}
          </section>
        ) : null}

        {toolCalls.length > 0 ? (
          <section className="chat-response-section">
            <h4>사용 툴</h4>
            <div className="chat-trace-list">
              {toolCalls.map((trace, index) => (
                <article
                  key={`${trace.name}-${index}`}
                  className="chat-trace-card"
                >
                  <div className="chat-trace-card-head">
                    <strong>{trace.label || trace.name}</strong>
                    {trace.status ? <span>{trace.status}</span> : null}
                  </div>
                  {renderSearchTraceChips(trace).length > 0 ? (
                    <div className="detail-chip-row">
                      {renderSearchTraceChips(trace).map((chip) => (
                        <span key={chip} className="detail-chip">
                          {chip}
                        </span>
                      ))}
                    </div>
                  ) : null}
                  {renderToolSummary(trace, documents) ? (
                    <p>{renderToolSummary(trace, documents)}</p>
                  ) : null}
                  {trace.rerank_error ? (
                    <p>리랭크 오류: {trace.rerank_error}</p>
                  ) : null}
                </article>
              ))}
            </div>
          </section>
        ) : showImplicitRetrievalNotice ? (
          <section className="chat-response-section">
            <h4>실행 방식</h4>
            <div className="chat-trace-list">
              <article className="chat-trace-card">
                <strong>내부 검색 경로</strong>
                <p>
                  명시적 툴 호출 없이 답변 흐름 안에서
                  {resolvedRetrievalMode ? ` ${resolvedRetrievalMode}` : ""} 검색으로
                  근거를 확보했습니다.
                  {evidenceChunks.length > 0 ? ` 사용된 청크는 ${evidenceChunks.length}개입니다.` : ""}
                </p>
              </article>
            </div>
          </section>
        ) : null}

        {logEntries.length > 0 ? (
          <section className="chat-response-section">
            <h4>진행 단계</h4>
            <ol className="chat-log-list">
              {logEntries.map((entry, index) => (
                <li key={`${entry}-${index}`}>{entry}</li>
              ))}
            </ol>
          </section>
        ) : null}

        {evidenceChunks.length > 0 ? (
          <section className="chat-response-section">
            <h4>검색에 사용된 청크</h4>
            <div className="chat-evidence-grid">
              {evidenceChunks.map((chunk, index) => (
                <article
                  key={`${chunk.chunk_id}-${index}`}
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
          </section>
        ) : null}
      </div>
    </details>
  );
}
