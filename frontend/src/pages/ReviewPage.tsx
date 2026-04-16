import { useEffect, useMemo, useState } from "react";
import { Link, useParams } from "react-router-dom";

import { PreviewPane } from "../components/PreviewPane";
import {
  applyReview,
  getDocument,
  getReviewSource,
  runStage,
  saveReviewDecisions,
} from "../lib/api";
import type {
  DocumentRecord,
  ReviewDecisionsPayload,
  ReviewElement,
  ReviewSourceResponse,
} from "../types";

type ReviewPhase = "edit" | "preview";

function applyLocalReviewState(
  elements: ReviewElement[],
  decisions: ReviewDecisionsPayload,
): ReviewElement[] {
  const exactTextDrop = new Set(decisions.exact_text_drop);
  return elements.map((element) => {
    const explicit = decisions.element_decisions[String(element.id)] || {};
    const explicitDropped = explicit.dropped;
    const droppedByExact = Boolean(
      element.normalized_text && exactTextDrop.has(element.normalized_text),
    );
    const dropped =
      explicitDropped !== undefined && explicitDropped !== null
        ? Boolean(explicitDropped)
        : droppedByExact;

    return {
      ...element,
      dropped,
      effective_category: explicit.category_override || element.category,
      drop_source:
        explicitDropped !== undefined && explicitDropped !== null
          ? "element"
          : droppedByExact
            ? "exact_text"
            : "none",
      review: {
        dropped: explicitDropped,
        category_override: explicit.category_override ?? null,
      },
    };
  });
}

function buildPreviewDecisions(
  decisions: ReviewDecisionsPayload,
  queuedDropIds: number[],
): ReviewDecisionsPayload {
  const nextElementDecisions = { ...decisions.element_decisions };
  queuedDropIds.forEach((elementId) => {
    nextElementDecisions[String(elementId)] = {
      ...nextElementDecisions[String(elementId)],
      dropped: true,
    };
  });
  return {
    ...decisions,
    element_decisions: nextElementDecisions,
  };
}

function getElementById(
  elements: ReviewElement[],
  elementId: number | null,
): ReviewElement | null {
  if (elementId === null) {
    return null;
  }
  return elements.find((element) => element.id === elementId) || null;
}

export function ReviewPage() {
  const { documentId = "" } = useParams();
  const [documentInfo, setDocumentInfo] = useState<DocumentRecord | null>(null);
  const [reviewSource, setReviewSource] = useState<ReviewSourceResponse | null>(null);
  const [decisions, setDecisions] = useState<ReviewDecisionsPayload>({
    element_decisions: {},
    exact_text_drop: [],
  });
  const [reviewPhase, setReviewPhase] = useState<ReviewPhase>("edit");
  const [queuedDropIds, setQueuedDropIds] = useState<number[]>([]);
  const [selectedElementId, setSelectedElementId] = useState<number | null>(null);
  const [busyAction, setBusyAction] = useState<string | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [successMessage, setSuccessMessage] = useState<string | null>(null);

  useEffect(() => {
    if (!documentId) {
      return;
    }

    async function load() {
      try {
        const [documentResponse, reviewResponse] = await Promise.all([
          getDocument(documentId),
          getReviewSource(documentId),
        ]);
        setDocumentInfo(documentResponse.document);
        setReviewSource(reviewResponse);
        setDecisions(reviewResponse.review_decisions);
        setQueuedDropIds([]);
        setReviewPhase("edit");
        setSelectedElementId(reviewResponse.elements[0]?.id ?? null);
      } catch (error) {
        setErrorMessage(
          error instanceof Error
            ? error.message
            : "review source를 불러오지 못했습니다.",
        );
      }
    }

    void load();
  }, [documentId]);

  const committedElements = useMemo(
    () => (reviewSource ? applyLocalReviewState(reviewSource.elements, decisions) : []),
    [reviewSource, decisions],
  );

  const previewElements = useMemo(() => {
    if (!reviewSource) {
      return [];
    }
    return applyLocalReviewState(
      reviewSource.elements,
      buildPreviewDecisions(decisions, queuedDropIds),
    );
  }, [decisions, queuedDropIds, reviewSource]);

  const activeElements = reviewPhase === "preview" ? previewElements : committedElements;
  const queuedElementSet = useMemo(() => new Set(queuedDropIds), [queuedDropIds]);
  const queuedElements = useMemo(
    () => committedElements.filter((element) => queuedElementSet.has(element.id)),
    [committedElements, queuedElementSet],
  );
  const droppedElements = useMemo(
    () => committedElements.filter((element) => element.dropped),
    [committedElements],
  );
  const selectedElement =
    getElementById(activeElements, selectedElementId) ||
    getElementById(committedElements, selectedElementId);

  const canPersist = reviewPhase === "edit" && queuedDropIds.length === 0;

  function setElementDecision(
    elementId: number,
    next: { dropped?: boolean | null; category_override?: string | null },
  ) {
    setDecisions((previous) => ({
      ...previous,
      element_decisions: {
        ...previous.element_decisions,
        [String(elementId)]: {
          ...previous.element_decisions[String(elementId)],
          ...next,
        },
      },
    }));
  }

  function toggleQueuedElement(elementId: number) {
    setSelectedElementId(elementId);

    const target = committedElements.find((element) => element.id === elementId);
    if (!target || target.dropped) {
      return;
    }

    setQueuedDropIds((previous) =>
      previous.includes(elementId)
        ? previous.filter((id) => id !== elementId)
        : [...previous, elementId],
    );
  }

  function removeFromQueue(elementId: number) {
    setQueuedDropIds((previous) => previous.filter((id) => id !== elementId));
  }

  function restoreElement(elementId: number) {
    setQueuedDropIds((previous) => previous.filter((id) => id !== elementId));
    setElementDecision(elementId, { dropped: false });
    setSelectedElementId(elementId);
  }

  function enterPreviewMode() {
    setErrorMessage(null);
    setSuccessMessage(null);
    if (queuedDropIds.length === 0) {
      setErrorMessage("장바구니에 담긴 요소가 없습니다.");
      return;
    }
    setReviewPhase("preview");
  }

  function returnToEditMode() {
    setReviewPhase("edit");
    setErrorMessage(null);
    setSuccessMessage(null);
  }

  function commitQueuedPreview() {
    if (queuedDropIds.length === 0) {
      setReviewPhase("edit");
      return;
    }

    setDecisions((previous) => {
      const nextElementDecisions = { ...previous.element_decisions };
      queuedDropIds.forEach((elementId) => {
        nextElementDecisions[String(elementId)] = {
          ...nextElementDecisions[String(elementId)],
          dropped: true,
        };
      });
      return {
        ...previous,
        element_decisions: nextElementDecisions,
      };
    });

    setSuccessMessage(`${queuedDropIds.length}개 요소를 drop 예정 목록에 반영했습니다.`);
    setQueuedDropIds([]);
    setReviewPhase("edit");
  }

  async function handleSave() {
    if (!documentId || !canPersist) {
      setErrorMessage("장바구니를 먼저 미리보기로 확인하고 확정한 뒤 저장하세요.");
      return;
    }
    setBusyAction("save");
    setErrorMessage(null);
    setSuccessMessage(null);
    try {
      await saveReviewDecisions(documentId, decisions);
      setSuccessMessage("review decisions를 저장했습니다.");
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : "저장에 실패했습니다.");
    } finally {
      setBusyAction(null);
    }
  }

  async function handleApply() {
    if (!documentId || !canPersist) {
      setErrorMessage("장바구니를 먼저 미리보기로 확인하고 확정한 뒤 반영하세요.");
      return;
    }
    setBusyAction("apply");
    setErrorMessage(null);
    setSuccessMessage(null);
    try {
      await saveReviewDecisions(documentId, decisions);
      await applyReview(documentId);
      const updatedDocument = await getDocument(documentId);
      setDocumentInfo(updatedDocument.document);
      setSuccessMessage("review overlay를 반영했습니다.");
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : "review 반영에 실패했습니다.");
    } finally {
      setBusyAction(null);
    }
  }

  async function handleRunStage3() {
    if (!documentId || !canPersist) {
      setErrorMessage("장바구니를 먼저 미리보기로 확인하고 확정한 뒤 다음 단계로 진행하세요.");
      return;
    }
    setBusyAction("stage3");
    setErrorMessage(null);
    setSuccessMessage(null);
    try {
      await saveReviewDecisions(documentId, decisions);
      await applyReview(documentId);
      await runStage(documentId, "stage3");
      const updatedDocument = await getDocument(documentId);
      setDocumentInfo(updatedDocument.document);
      setSuccessMessage("stage3까지 진행했습니다.");
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : "stage3 실행에 실패했습니다.");
    } finally {
      setBusyAction(null);
    }
  }

  if (!reviewSource) {
    return (
      <div className="app-page">
        <section className="section-card">
          <p>review source를 불러오는 중입니다...</p>
          {errorMessage ? <div className="error-banner">{errorMessage}</div> : null}
        </section>
      </div>
    );
  }

  return (
    <div className="review-page-shell">
      <header className="review-topbar">
        <div>
          <Link className="back-link" to="/">
            문서 목록으로
          </Link>
          <h1>{documentInfo?.original_filename || reviewSource.document_id}</h1>
          <p className="muted-text">
            가운데 문서형 preview에서 요소를 한 번 클릭하면 장바구니에 담깁니다. 제거 미리보기를 확인한 뒤 확정하면 됩니다.
          </p>
        </div>
        <div className="review-topbar-actions">
          {reviewPhase === "edit" ? (
            <>
              <button className="ghost-button" onClick={() => void handleSave()} disabled={!canPersist}>
                {busyAction === "save" ? "저장 중..." : "결정 저장"}
              </button>
              <button className="secondary-button" onClick={() => void handleApply()} disabled={!canPersist}>
                {busyAction === "apply" ? "반영 중..." : "검수 반영"}
              </button>
              <button className="primary-button" onClick={() => void handleRunStage3()} disabled={!canPersist}>
                {busyAction === "stage3" ? "진행 중..." : "다음 단계 진행"}
              </button>
            </>
          ) : (
            <>
              <button className="ghost-button" onClick={returnToEditMode}>
                이전 편집으로 돌아가기
              </button>
              <button className="primary-button" onClick={commitQueuedPreview}>
                이 미리보기 확정
              </button>
            </>
          )}
        </div>
      </header>

      {errorMessage ? <div className="error-banner">{errorMessage}</div> : null}
      {successMessage ? <div className="success-banner">{successMessage}</div> : null}

      <div className="review-layout">
        <PreviewPane
          documentId={documentId}
          elements={activeElements}
          mode={reviewPhase}
          queuedElementIds={queuedDropIds}
          selectedElementId={selectedElementId}
          onElementClick={(elementId) => {
            if (reviewPhase === "edit") {
              toggleQueuedElement(elementId);
              return;
            }
            setSelectedElementId(elementId);
          }}
        />

        <aside className="review-sidebar review-sidebar--cart">
          <section className="sidebar-panel">
            <div className="sidebar-panel-header">
              <h2>검수 흐름</h2>
              <p>{reviewPhase === "preview" ? "drop 미리보기" : "편집 중"}</p>
            </div>
            <div className="review-flow-stats">
              <div className="review-stat-card">
                <strong>{reviewSource.counts.total_elements}</strong>
                <span>전체 요소</span>
              </div>
              <div className="review-stat-card">
                <strong>{droppedElements.length}</strong>
                <span>현재 drop 예정</span>
              </div>
              <div className="review-stat-card is-accent">
                <strong>{queuedElements.length}</strong>
                <span>장바구니</span>
              </div>
            </div>
            <p className="panel-note">
              {reviewPhase === "edit"
                ? "preview에서 hover 시 요소 경계가 보이고, 클릭하면 바로 장바구니에 담깁니다."
                : "현재는 장바구니 요소를 제거한 결과만 보여줍니다. 마음에 들지 않으면 바로 이전 편집 상태로 돌아갈 수 있습니다."}
            </p>
            {reviewPhase === "edit" ? (
              <div className="detail-actions">
                <button
                  className="secondary-button"
                  onClick={enterPreviewMode}
                  disabled={queuedElements.length === 0}
                >
                  선택 항목 제거 미리보기
                </button>
                <button
                  className="ghost-button"
                  onClick={() => setQueuedDropIds([])}
                  disabled={queuedElements.length === 0}
                >
                  장바구니 비우기
                </button>
              </div>
            ) : (
              <div className="detail-actions">
                <button className="ghost-button" onClick={returnToEditMode}>
                  이전 편집으로 돌아가기
                </button>
                <button className="primary-button" onClick={commitQueuedPreview}>
                  이 미리보기 확정
                </button>
              </div>
            )}
          </section>

          <section className="sidebar-panel list-panel">
            <div className="sidebar-panel-header">
              <h2>{reviewPhase === "preview" ? "이번 미리보기에서 제거될 요소" : "장바구니"}</h2>
              <p>{queuedElements.length}개</p>
            </div>
            <div className="cart-list">
              {queuedElements.length === 0 ? (
                <p className="muted-text">아직 장바구니에 담긴 요소가 없습니다.</p>
              ) : null}
              {queuedElements.map((element) => (
                <div key={element.id} className="cart-item">
                  <button
                    className="cart-item-body"
                    onClick={() => setSelectedElementId(element.id)}
                  >
                    <div className="element-list-item-top">
                      <span>#{element.id}</span>
                      <span>p.{element.page ?? "-"}</span>
                      <span>{element.effective_category}</span>
                    </div>
                    <div className="element-list-item-body">
                      {String(element.text || element.html || "").slice(0, 120) || "(empty)"}
                    </div>
                  </button>
                  <button
                    className="ghost-button cart-item-action"
                    onClick={() => removeFromQueue(element.id)}
                  >
                    제거
                  </button>
                </div>
              ))}
            </div>

            <div className="sidebar-subsection">
              <div className="sidebar-panel-header">
                <h3>이미 drop 예정인 요소</h3>
                <p>{droppedElements.length}개</p>
              </div>
              <div className="cart-list cart-list--compact">
                {droppedElements.length === 0 ? (
                  <p className="muted-text">아직 확정된 drop 요소가 없습니다.</p>
                ) : null}
                {droppedElements.map((element) => (
                  <div key={element.id} className="cart-item cart-item--compact">
                    <button
                      className="cart-item-body"
                      onClick={() => setSelectedElementId(element.id)}
                    >
                      <div className="element-list-item-top">
                        <span>#{element.id}</span>
                        <span>p.{element.page ?? "-"}</span>
                        <span>{element.effective_category}</span>
                      </div>
                      <div className="element-list-item-body">
                        {String(element.text || element.html || "").slice(0, 96) || "(empty)"}
                      </div>
                    </button>
                    <button
                      className="ghost-button cart-item-action"
                      onClick={() => restoreElement(element.id)}
                      disabled={reviewPhase === "preview"}
                    >
                      restore
                    </button>
                  </div>
                ))}
              </div>
            </div>
          </section>

          <section className="sidebar-panel detail-panel">
            <div className="sidebar-panel-header">
              <h2>선택 요소</h2>
              <p>{selectedElement ? `#${selectedElement.id}` : "선택 없음"}</p>
            </div>
            {selectedElement ? (
              <div className="detail-content">
                <div className="detail-chip-row">
                  <span className="detail-chip">page {selectedElement.page ?? "-"}</span>
                  <span className="detail-chip">raw {selectedElement.category}</span>
                  <span className="detail-chip">effective {selectedElement.effective_category}</span>
                  {selectedElement.dropped ? (
                    <span className="detail-chip is-danger">drop 예정</span>
                  ) : null}
                  {queuedElementSet.has(selectedElement.id) ? (
                    <span className="detail-chip is-queued">장바구니</span>
                  ) : null}
                </div>

                <div className="detail-actions">
                  <button
                    className={queuedElementSet.has(selectedElement.id) ? "primary-button" : "secondary-button"}
                    onClick={() => toggleQueuedElement(selectedElement.id)}
                    disabled={reviewPhase === "preview" || selectedElement.dropped}
                  >
                    {queuedElementSet.has(selectedElement.id) ? "장바구니 해제" : "장바구니 담기"}
                  </button>
                  <button
                    className="ghost-button"
                    onClick={() => restoreElement(selectedElement.id)}
                    disabled={reviewPhase === "preview"}
                  >
                    restore
                  </button>
                  <button
                    className="ghost-button"
                    disabled={reviewPhase === "preview" || !selectedElement.normalized_text}
                    onClick={() => {
                      if (!selectedElement.normalized_text) {
                        return;
                      }
                      setDecisions((previous) => ({
                        ...previous,
                        exact_text_drop: Array.from(
                          new Set([
                            ...previous.exact_text_drop,
                            selectedElement.normalized_text,
                          ]),
                        ),
                      }));
                    }}
                  >
                    exact match 전체 drop
                  </button>
                </div>

                <label className="field-group">
                  <span>category override</span>
                  <select
                    className="input"
                    value={
                      decisions.element_decisions[String(selectedElement.id)]
                        ?.category_override ?? ""
                    }
                    onChange={(event) =>
                      setElementDecision(selectedElement.id, {
                        category_override: event.target.value || null,
                      })
                    }
                    disabled={reviewPhase === "preview"}
                  >
                    <option value="">(없음)</option>
                    {reviewSource.allowed_category_overrides.map((category) => (
                      <option key={category} value={category}>
                        {category}
                      </option>
                    ))}
                  </select>
                </label>

                <div className="detail-section">
                  <h3>text</h3>
                  <pre>{String(selectedElement.text || "(empty)")}</pre>
                </div>
                <div className="detail-section">
                  <h3>prev / next context</h3>
                  <p><strong>prev</strong>: {selectedElement.prev_context || "(없음)"}</p>
                  <p><strong>next</strong>: {selectedElement.next_context || "(없음)"}</p>
                </div>
                <div className="detail-section">
                  <h3>exact match</h3>
                  <p>same_text_count: {selectedElement.same_text_count}</p>
                  <p>normalized_text: {selectedElement.normalized_text || "(없음)"}</p>
                </div>
              </div>
            ) : (
              <p className="muted-text">preview에서 요소를 클릭하면 선택됩니다.</p>
            )}
          </section>
        </aside>
      </div>
    </div>
  );
}
