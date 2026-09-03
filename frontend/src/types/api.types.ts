// API Request Types
export interface ChatRequest {
  question: string;
  conversation_id?: string;
  document_ids?: string[];
}

// API Response Types
export interface SourceDocument {
  content: string;
  document_name: string;
  document_id: string;
  page?: number;
  chunk_index: number;
  similarity_score: number | null;
  image_urls: string[];
}

export interface TokenUsage {
  input_tokens: number;
  output_tokens: number;
  total_tokens: number;
}

export interface ChatResponse {
  answer: string;
  sources: SourceDocument[];
  conversation_id: string;
  message_id: string;
  timestamp: string;
  token_usage: TokenUsage;
}

export interface UploadResponse {
  document_id: string;
  filename: string;
  num_chunks: number;
  status: string;
  message: string;
  chunking_strategy: string;
}

export type ChunkingStrategy = 'default' | 'semantic';

export interface UploadOptions {
  chunkingStrategy: ChunkingStrategy;
  /** "default" 전략에서만 쓰인다. 값을 주지 않으면 서버 기본값을 쓴다. */
  chunkSize?: number;
  chunkOverlap?: number;
}

export interface DocumentInfo {
  document_id: string;
  filename: string;
  upload_date: string;
  num_chunks: number;
  file_size: number;
}

export interface ParsedBlock {
  type: string;
  text?: string;
  table_body?: string;
  image_id?: string;
}

export interface ParsedPage {
  page_number: number;
  blocks: ParsedBlock[];
}

export interface ParsedDocumentSummary {
  document_id: string;
  filename: string;
  created_at: string;
  page_count: number;
}

export interface ParsedDocumentDetail extends ParsedDocumentSummary {
  pages: ParsedPage[];
}
