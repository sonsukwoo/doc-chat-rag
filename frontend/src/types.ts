export type StageStatus =
  | "not_started"
  | "uploaded"
  | "running"
  | "completed"
  | "failed";

export interface DocumentStageRecord {
  status?: StageStatus;
  updated_at?: string;
  error?: string | null;
  outputs?: Record<string, string>;
}

export interface DocumentRecord {
  document_id: string;
  original_filename: string;
  uploaded_at: string;
  stages: Record<string, DocumentStageRecord>;
}

export interface DocumentInfoResponse {
  document: DocumentRecord;
  paths: Record<string, string>;
}

export interface ThreadRecord {
  thread_id: string;
  thread_name: string;
  collection_name: string;
  description?: string | null;
  default_retrieval_mode: "dense" | "hybrid" | string;
  metadata: Record<string, unknown>;
  active_document_ids: string[];
  document_count: number;
  created_at?: string | null;
  updated_at?: string | null;
  archived_at?: string | null;
}

export interface ThreadDocumentRecord {
  document_id: string;
  original_filename: string;
  uploaded_at?: string | null;
  stages: Record<string, DocumentStageRecord>;
  source_pdf_path: string;
}

export interface ThreadDocumentPipelineResponse {
  thread: ThreadRecord;
  document: DocumentRecord;
  stage_status: Record<string, string>;
  review?: Record<string, unknown>;
  indexing?: Record<string, unknown>;
  next_step: string;
}

export interface ChatCitation {
  document_id?: string;
  chunk_id?: string;
  parent_id?: string | null;
  page?: number | null;
  section_title?: string | null;
  asset_ref?: string | null;
  asset_relative_path?: string | null;
}

export interface ChatVisualAsset {
  asset_ref: string;
  document_id: string;
  chunk_id: string;
  asset_kind: string;
  relative_path: string;
  asset_stage: "source" | "stage1" | "stage2" | "review" | "stage3" | "stage4";
  page?: number | null;
  caption?: string | null;
  summary_text?: string | null;
  heading_path?: string[];
  pages?: number[];
}

export interface ChatEvidenceChunk {
  document_id: string;
  chunk_id: string;
  parent_id?: string | null;
  page?: number | null;
  section_title?: string | null;
  chunk_type?: string | null;
  text_excerpt: string;
}

export interface ChatInterruptPayload {
  kind?: "clarification";
  question?: string;
  reason?: string;
  options?: string[];
}

export interface ThreadChatHistoryMessage {
  role: "user" | "assistant";
  content: string;
  kind: "answer" | "interrupt";
}

export interface ThreadChatTurnResult {
  status: "completed" | "interrupted";
  thread_id: string;
  final_answer?: string | null;
  citations: ChatCitation[];
  visual_assets: ChatVisualAsset[];
  evidence_chunks: ChatEvidenceChunk[];
  interrupt?: ChatInterruptPayload | null;
  retrieval_mode?: string;
  logs?: string[];
}

export interface ReviewElement {
  id: number;
  page?: number;
  category: string;
  effective_category: string;
  text?: string;
  html?: string;
  image_path?: string;
  bbox?: number[];
  table_summary?: string;
  visual_summary?: string;
  resolved_caption?: string;
  internal_caption_text?: string;
  normalized_text: string;
  same_text_count: number;
  dropped: boolean;
  drop_source: "element" | "exact_text" | "none";
  review: {
    dropped?: boolean | null;
    category_override?: string | null;
  };
  prev_context: string;
  next_context: string;
  [key: string]: unknown;
}

export interface ReviewDecisionsPayload {
  element_decisions: Record<
    string,
    {
      dropped?: boolean | null;
      category_override?: string | null;
    }
  >;
  exact_text_drop: string[];
}

export interface ReviewSourceResponse {
  document_id: string;
  source_pdf?: string;
  total_pages?: number;
  document_profile?: Record<string, unknown>;
  allowed_category_overrides: string[];
  review_decisions: ReviewDecisionsPayload;
  counts: {
    total_elements: number;
    dropped_elements: number;
  };
  elements: ReviewElement[];
}

export interface ThreadChatBootstrapResponse {
  thread: ThreadRecord;
}

export interface ThreadChatViewResponse {
  thread: ThreadRecord;
  messages: ThreadChatHistoryMessage[];
  interrupt?: ChatInterruptPayload | null;
  history_notice?: string | null;
}

export interface ThreadDeleteResponse {
  status: "deleted";
  thread_id: string;
  deleted_document_ids: string[];
  deleted_checkpoint_rows: Record<string, number>;
  deleted_collection_name?: string | null;
  cleanup_warnings: string[];
}
