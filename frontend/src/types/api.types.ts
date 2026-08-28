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
  similarity_score?: number;
  image_urls: string[];
}

export interface ChatResponse {
  answer: string;
  sources: SourceDocument[];
  conversation_id: string;
  message_id: string;
  timestamp: string;
}

export interface UploadResponse {
  document_id: string;
  filename: string;
  num_chunks: number;
  status: string;
  message: string;
}

export interface DocumentInfo {
  document_id: string;
  filename: string;
  upload_date: string;
  num_chunks: number;
  file_size: number;
}
