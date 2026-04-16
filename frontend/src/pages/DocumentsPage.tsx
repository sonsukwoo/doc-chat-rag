import { useEffect, useMemo, useState } from "react";
import type { DragEvent } from "react";
import { Link } from "react-router-dom";

import { getDocument, listDocuments, runStage, uploadDocument } from "../lib/api";
import { StageBadge } from "../components/StageBadge";
import type { DocumentRecord } from "../types";

type BusyMap = Record<string, string | null>;

export function DocumentsPage() {
  const [documents, setDocuments] = useState<DocumentRecord[]>([]);
  const [uploading, setUploading] = useState(false);
  const [busyMap, setBusyMap] = useState<BusyMap>({});
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  async function refreshDocuments() {
    const response = await listDocuments();
    setDocuments(response.documents);
  }

  useEffect(() => {
    void refreshDocuments();
  }, []);

  async function handleFileUpload(file: File) {
    setErrorMessage(null);
    setUploading(true);
    try {
      const response = await uploadDocument(file);
      setDocuments((previous) => [response.document, ...previous]);
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : "업로드에 실패했습니다.");
    } finally {
      setUploading(false);
    }
  }

  async function handleStageRun(documentId: string, stage: "stage1" | "stage2" | "stage3") {
    setErrorMessage(null);
    setBusyMap((previous) => ({ ...previous, [documentId]: stage }));
    try {
      await runStage(documentId, stage);
      const updated = await getDocument(documentId);
      setDocuments((previous) =>
        previous.map((item) =>
          item.document_id === documentId ? updated.document : item,
        ),
      );
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : `${stage} 실행에 실패했습니다.`);
    } finally {
      setBusyMap((previous) => ({ ...previous, [documentId]: null }));
    }
  }

  function onDrop(event: DragEvent<HTMLLabelElement>) {
    event.preventDefault();
    const file = event.dataTransfer.files?.[0];
    if (file) {
      void handleFileUpload(file);
    }
  }

  const sortedDocuments = useMemo(
    () =>
      [...documents].sort((a, b) =>
        (b.uploaded_at || "").localeCompare(a.uploaded_at || ""),
      ),
    [documents],
  );

  return (
    <div className="app-page">
      <section className="hero-card">
        <div className="hero-copy">
          <p className="eyebrow">Local Review Workspace</p>
          <h1>문서 업로드 후 stage별로 처리하고, stage2 결과를 바로 검수합니다.</h1>
          <p className="hero-description">
            현재 버전은 업로드, stage1/stage2/stage3 실행, review 진입까지를 로컬 웹앱으로 묶습니다.
          </p>
        </div>
        <label
          className={`upload-dropzone ${uploading ? "is-busy" : ""}`}
          onDragOver={(event) => event.preventDefault()}
          onDrop={onDrop}
        >
          <input
            type="file"
            accept="application/pdf"
            onChange={(event) => {
              const file = event.target.files?.[0];
              if (file) {
                void handleFileUpload(file);
              }
            }}
          />
          <div className="upload-dropzone-content">
            <strong>{uploading ? "업로드 중..." : "PDF를 드래그해서 올리거나 선택하세요"}</strong>
            <span>원본 파일명은 보존하고, 내부 저장은 document_id 기준으로 진행합니다.</span>
          </div>
        </label>
      </section>

      {errorMessage ? <div className="error-banner">{errorMessage}</div> : null}

      <section className="section-card">
        <div className="section-card-header">
          <div>
            <p className="eyebrow">Documents</p>
            <h2>문서 목록</h2>
          </div>
          <button className="ghost-button" onClick={() => void refreshDocuments()}>
            새로고침
          </button>
        </div>

        <div className="documents-grid">
          {sortedDocuments.length === 0 ? (
            <div className="empty-state">
              아직 업로드된 문서가 없습니다. PDF를 올린 뒤 stage를 순차 실행하세요.
            </div>
          ) : null}

          {sortedDocuments.map((document) => {
            const busyStage = busyMap[document.document_id];
            const stages = document.stages || {};
            const stage1Status = stages.stage1?.status;
            const stage2Status = stages.stage2?.status;
            const reviewStatus = stages.review?.status;
            const stage3Status = stages.stage3?.status;

            return (
              <article key={document.document_id} className="document-card">
                <div className="document-card-header">
                  <div>
                    <h3>{document.original_filename}</h3>
                    <p className="muted-text">{document.document_id}</p>
                  </div>
                  <span className="timestamp">{document.uploaded_at}</span>
                </div>

                <div className="document-stage-list">
                  <StageBadge label="upload" status={stages.upload?.status} />
                  <StageBadge label="stage1" status={stage1Status} />
                  <StageBadge label="stage2" status={stage2Status} />
                  <StageBadge label="review" status={reviewStatus} />
                  <StageBadge label="stage3" status={stage3Status} />
                </div>

                <div className="document-actions">
                  <button
                    className="primary-button"
                    disabled={busyStage !== undefined && busyStage !== null}
                    onClick={() => void handleStageRun(document.document_id, "stage1")}
                  >
                    {busyStage === "stage1" ? "Stage1 실행 중..." : "Stage1 실행"}
                  </button>
                  <button
                    className="secondary-button"
                    disabled={stage1Status !== "completed" || Boolean(busyStage)}
                    onClick={() => void handleStageRun(document.document_id, "stage2")}
                  >
                    {busyStage === "stage2" ? "Stage2 실행 중..." : "Stage2 실행"}
                  </button>
                  <Link
                    className={`secondary-button link-button ${
                      stage2Status !== "completed" ? "is-disabled" : ""
                    }`}
                    to={
                      stage2Status === "completed"
                        ? `/documents/${document.document_id}/review`
                        : "#"
                    }
                    onClick={(event) => {
                      if (stage2Status !== "completed") {
                        event.preventDefault();
                      }
                    }}
                  >
                    검수 열기
                  </Link>
                  <button
                    className="secondary-button"
                    disabled={stage2Status !== "completed" || Boolean(busyStage)}
                    onClick={() => void handleStageRun(document.document_id, "stage3")}
                  >
                    {busyStage === "stage3" ? "Stage3 실행 중..." : "Stage3 실행"}
                  </button>
                </div>
              </article>
            );
          })}
        </div>
      </section>
    </div>
  );
}
