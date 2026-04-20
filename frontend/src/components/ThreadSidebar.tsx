import type { ThreadRecord } from "../types";

type ThreadSidebarProps = {
  threads: ThreadRecord[];
  selectedThreadId: string | null;
  onSelectThread: (threadId: string) => void;
  onCreateThread: () => void;
  onDeleteThread?: (threadId: string) => void;
  deletingThreadId?: string | null;
};

function ThreadDeleteIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path
        d="M9 4.5h6m-8 3h10m-8.5 0v9.25a1.25 1.25 0 0 0 1.25 1.25h4.5a1.25 1.25 0 0 0 1.25-1.25V7.5M10 10.5v4.5m4-4.5v4.5"
        fill="none"
        stroke="currentColor"
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth="1.8"
      />
    </svg>
  );
}

export function ThreadSidebar({
  threads,
  selectedThreadId,
  onSelectThread,
  onCreateThread,
  onDeleteThread,
  deletingThreadId,
}: ThreadSidebarProps) {
  return (
    <aside className="thread-sidebar">
      <div className="thread-sidebar-brand">
        <div className="workspace-brand-mark">DC</div>
        <div>
          <p className="eyebrow">Doc Chat</p>
        </div>
      </div>

      <button className="thread-create-button" onClick={onCreateThread}>
        + 새 채팅방
      </button>

      <section className="thread-sidebar-section">
        <div className="thread-sidebar-section-header">
          <div>
            <p className="eyebrow">Threads</p>
            <h2>채팅방</h2>
          </div>
        </div>

        <div className="thread-list">
          {threads.length === 0 ? (
            <div className="empty-state thread-sidebar-empty">
              아직 생성된 채팅방이 없습니다.
            </div>
          ) : null}

          {threads.map((thread) => {
            const isActive = thread.thread_id === selectedThreadId;
            const isDeleting = deletingThreadId === thread.thread_id;
            return (
              <article
                key={thread.thread_id}
                className={`thread-list-item ${isActive ? "is-active" : ""}`}
              >
                <button
                  type="button"
                  className="thread-list-item-button"
                  onClick={() => onSelectThread(thread.thread_id)}
                >
                  <div className="thread-list-item-top">
                    <strong>{thread.thread_name}</strong>
                  </div>
                  <div className="thread-list-item-meta">
                    <span>{thread.document_count} docs</span>
                    <span>설정 {thread.default_retrieval_mode}</span>
                  </div>
                </button>

                {onDeleteThread ? (
                  <button
                    type="button"
                    className="thread-list-delete-button"
                    aria-label={`${thread.thread_name} 삭제`}
                    title={`${thread.thread_name} 삭제`}
                    onClick={() => onDeleteThread(thread.thread_id)}
                    disabled={isDeleting}
                  >
                    <ThreadDeleteIcon />
                  </button>
                ) : null}
              </article>
            );
          })}
        </div>
      </section>
    </aside>
  );
}
