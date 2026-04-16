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
  allowed_category_overrides: string[];
  review_decisions: ReviewDecisionsPayload;
  counts: {
    total_elements: number;
    dropped_elements: number;
  };
  elements: ReviewElement[];
}
