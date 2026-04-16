import type { ReviewElement } from "../types";
import { buildStageAssetUrl } from "../lib/api";

interface PreviewPaneProps {
  documentId: string;
  elements: ReviewElement[];
  mode: "edit" | "preview";
  queuedElementIds: number[];
  selectedElementId: number | null;
  onElementClick: (elementId: number) => void;
}

function patchRelativeAssetPaths(
  html: string,
  documentId: string,
  stage: "stage2" | "review",
): string {
  return html.replace(
    /(src|href)=["'](?!https?:\/\/|data:|\/)([^"']+)["']/g,
    (_match, attr, relativePath) =>
      `${attr}="${buildStageAssetUrl(documentId, stage, relativePath)}"`,
  );
}

function renderTextElement(
  documentId: string,
  element: ReviewElement,
): JSX.Element | null {
  const category = element.effective_category;
  const text = String(element.text || "").trim();
  const html = String(element.html || "").trim();

  if (category === "heading") {
    return <h2>{text || html}</h2>;
  }
  if (category === "code") {
    return <pre>{text}</pre>;
  }
  if (category === "list") {
    const lines = text
      .split(/\n+/)
      .map((item) => item.trim())
      .filter(Boolean);
    return (
      <ul>
        {lines.map((item) => (
          <li key={item}>{item}</li>
        ))}
      </ul>
    );
  }
  if (html) {
    return (
      <div
        dangerouslySetInnerHTML={{
          __html: patchRelativeAssetPaths(html, documentId, "stage2"),
        }}
      />
    );
  }
  if (!text) {
    return null;
  }
  return <p>{text}</p>;
}

function getCaption(element: ReviewElement): string {
  return String(element.resolved_caption || element.internal_caption_text || "").trim();
}

function PreviewBlock({
  documentId,
  element,
}: {
  documentId: string;
  element: ReviewElement;
}) {
  const category = element.effective_category;
  const caption = getCaption(element);
  const imagePath = typeof element.image_path === "string" ? element.image_path : "";

  if (category === "figure") {
    return (
      <figure className="preview-figure">
        {imagePath ? (
          <img
            src={buildStageAssetUrl(documentId, "stage2", imagePath)}
            alt={caption || "figure"}
          />
        ) : null}
        {caption ? <figcaption>{caption}</figcaption> : null}
        {element.visual_summary ? (
          <p className="preview-summary">요약: {String(element.visual_summary)}</p>
        ) : null}
      </figure>
    );
  }

  if (category === "table") {
    return (
      <section className="preview-table">
        {caption ? <p className="preview-caption">{caption}</p> : null}
        {element.html ? (
          <div
            dangerouslySetInnerHTML={{
              __html: patchRelativeAssetPaths(String(element.html), documentId, "stage2"),
            }}
          />
        ) : (
          <pre>{String(element.text || "")}</pre>
        )}
        {element.table_summary ? (
          <p className="preview-summary">요약: {String(element.table_summary)}</p>
        ) : null}
      </section>
    );
  }

  return renderTextElement(documentId, element);
}

export function PreviewPane({
  documentId,
  elements,
  mode,
  queuedElementIds,
  selectedElementId,
  onElementClick,
}: PreviewPaneProps) {
  const queuedSet = new Set(queuedElementIds);
  const visibleElements =
    mode === "preview" ? elements.filter((element) => !element.dropped) : elements;

  return (
    <div className="preview-pane">
      <div className="preview-pane-header">
        <div>
          <h2>Document Preview</h2>
          <p>
            {mode === "edit"
              ? "마우스를 올리면 요소 경계가 보입니다. 클릭 한 번으로 장바구니에 담습니다."
              : "장바구니 요소를 제거한 결과를 문서 형태로 미리 봅니다."}
          </p>
        </div>
        <div className="preview-pane-badges">
          <span className="detail-chip">{visibleElements.length}개 표시</span>
          <span className="detail-chip is-queued">{queuedElementIds.length}개 장바구니</span>
        </div>
      </div>
      <div className="preview-scroll">
        {visibleElements.map((element) => {
          const isQueued = queuedSet.has(element.id);
          const classes = [
            "preview-block",
            mode === "edit" ? "is-editable" : "is-preview",
            selectedElementId === element.id ? "is-selected" : "",
            isQueued ? "is-queued" : "",
            element.dropped ? "is-dropped" : "",
          ]
            .filter(Boolean)
            .join(" ");

          return (
            <article
              key={element.id}
              className={classes}
              onClick={() => onElementClick(element.id)}
            >
              <div className="preview-block-meta">
                <span>#{element.id}</span>
                <span>p.{element.page ?? "-"}</span>
                <span>{element.effective_category}</span>
              </div>
              <div className="preview-block-status">
                {isQueued ? <span className="preview-status-chip is-queued">장바구니</span> : null}
                {element.dropped ? (
                  <span className="preview-status-chip is-dropped">drop 예정</span>
                ) : null}
              </div>
              <PreviewBlock documentId={documentId} element={element} />
            </article>
          );
        })}
      </div>
    </div>
  );
}
