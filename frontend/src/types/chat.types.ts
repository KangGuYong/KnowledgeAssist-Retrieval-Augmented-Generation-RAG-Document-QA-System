import { SourceDocument } from './api.types';

export interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  sources?: SourceDocument[];
  timestamp: Date;
  isLoading?: boolean;
}

export interface Conversation {
  id: string;
  messages: Message[];
  createdAt: Date;
}

export interface UploadedFile {
  id: string;
  name: string;
  size: number;
  status: 'uploading' | 'success' | 'error';
  numChunks?: number;
  errorMessage?: string;
}
