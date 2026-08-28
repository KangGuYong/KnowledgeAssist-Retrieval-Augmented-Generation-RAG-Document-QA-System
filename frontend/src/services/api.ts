import axios, { AxiosInstance, AxiosError } from 'axios';
import { ChatRequest, ChatResponse, UploadResponse, UploadOptions, DocumentInfo } from '../types/api.types';

function appendUploadOptions(formData: FormData, options?: UploadOptions): void {
  if (!options) return;
  formData.append('chunking_strategy', options.chunkingStrategy);
  if (options.chunkSize !== undefined) {
    formData.append('chunk_size', String(options.chunkSize));
  }
  if (options.chunkOverlap !== undefined) {
    formData.append('chunk_overlap', String(options.chunkOverlap));
  }
}

// Empty by default so requests go to the same origin and Vite's /api proxy
// forwards them to the backend (no CORS involved). Set VITE_API_BASE_URL only
// when the backend lives on a different origin than the frontend.
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? '';
const API_PREFIX = '/api/v1';

// Quick metadata calls (list/delete) should fail fast.
const DEFAULT_TIMEOUT = 30_000;
// Chat and upload wait on the LLM and the embedding model, which routinely take
// tens of seconds and can cold-start much slower.
const LLM_TIMEOUT = 300_000;

class ApiService {
  private client: AxiosInstance;

  constructor() {
    this.client = axios.create({
      baseURL: `${API_BASE_URL}${API_PREFIX}`,
      headers: {
        'Content-Type': 'application/json',
      },
      timeout: DEFAULT_TIMEOUT,
    });

    // Response interceptor for error handling
    this.client.interceptors.response.use(
      (response) => response,
      (error: AxiosError) => {
        if (error.response) {
          console.error('API Error:', error.response.data);
          throw new Error(
            (error.response.data as any)?.detail || 'An error occurred'
          );
        } else if (error.code === 'ECONNABORTED' || error.code === 'ETIMEDOUT') {
          throw new Error(
            'The server took too long to respond. The model may still be working — try again, or ask a shorter question.'
          );
        } else if (error.request) {
          throw new Error('No response from server. Please check your connection.');
        } else {
          throw new Error(error.message);
        }
      }
    );
  }

  /**
   * Upload a single file
   */
  async uploadFile(file: File, options?: UploadOptions): Promise<UploadResponse> {
    const formData = new FormData();
    formData.append('file', file);
    appendUploadOptions(formData, options);

    const response = await this.client.post<UploadResponse>(
      '/upload/',
      formData,
      {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        timeout: LLM_TIMEOUT,
      }
    );

    return response.data;
  }

  /**
   * Upload multiple files. options apply to every file in the batch.
   */
  async uploadFiles(files: File[], options?: UploadOptions): Promise<UploadResponse[]> {
    const formData = new FormData();
    files.forEach((file) => {
      formData.append('files', file);
    });
    appendUploadOptions(formData, options);

    const response = await this.client.post<UploadResponse[]>(
      '/upload/batch',
      formData,
      {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        timeout: LLM_TIMEOUT,
      }
    );

    return response.data;
  }

  /**
   * Send a chat message
   */
  async sendMessage(request: ChatRequest): Promise<ChatResponse> {
    const response = await this.client.post<ChatResponse>('/chat/', request, {
      timeout: LLM_TIMEOUT,
    });
    return response.data;
  }

  /**
   * Clear conversation history
   */
  async clearConversation(conversationId: string): Promise<void> {
    await this.client.delete(`/chat/conversation/${conversationId}`);
  }

  /**
   * Get list of uploaded documents
   */
  async getDocuments(): Promise<DocumentInfo[]> {
    const response = await this.client.get<DocumentInfo[]>('/documents/');
    return response.data;
  }

  /**
   * Delete a document
   */
  async deleteDocument(documentId: string): Promise<void> {
    await this.client.delete(`/documents/${documentId}`);
  }
}

export const apiService = new ApiService();
