import React, { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { Upload, FileText, X, CheckCircle, AlertCircle, ChevronDown, ChevronUp } from 'lucide-react';
import { apiService } from '../services/api';
import { UploadedFile } from '../types/chat.types';
import { ChunkingStrategy, UploadOptions } from '../types/api.types';
import '../styles/FileUploader.css';

// 이보다 긴 파일명은 목록에서 잘라 표시한다.
const MAX_FILE_NAME_LENGTH = 15;

interface FileUploaderProps {
  onUploadComplete?: (documentIds: string[]) => void;
  maxFiles?: number;
}

export const FileUploader: React.FC<FileUploaderProps> = ({
  onUploadComplete,
  maxFiles = 10,
}) => {
  const [uploadedFiles, setUploadedFiles] = useState<UploadedFile[]>([]);
  const [isUploading, setIsUploading] = useState(false);

  const [showAdvanced, setShowAdvanced] = useState(false);
  const [chunkingStrategy, setChunkingStrategy] = useState<ChunkingStrategy>('default');
  const [chunkSize, setChunkSize] = useState('');
  const [chunkOverlap, setChunkOverlap] = useState('');

  const onDrop = useCallback(
    async (acceptedFiles: File[]) => {
      if (acceptedFiles.length === 0) return;

      setIsUploading(true);

      const options: UploadOptions = {
        chunkingStrategy,
        ...(chunkingStrategy === 'default' && chunkSize !== '' ? { chunkSize: Number(chunkSize) } : {}),
        ...(chunkingStrategy === 'default' && chunkOverlap !== '' ? { chunkOverlap: Number(chunkOverlap) } : {}),
      };

      // Add files to state with uploading status
      const newFiles: UploadedFile[] = acceptedFiles.map((file) => ({
        id: `temp-${Date.now()}-${file.name}`,
        name: file.name,
        size: file.size,
        status: 'uploading',
      }));

      setUploadedFiles((prev) => [...prev, ...newFiles]);

      // Upload files one by one (or in parallel if you prefer)
      const uploadedDocIds: string[] = [];

      for (let i = 0; i < acceptedFiles.length; i++) {
        const file = acceptedFiles[i];
        const tempId = newFiles[i].id;

        try {
          const response = await apiService.uploadFile(file, options);

          // Update file status to success
          setUploadedFiles((prev) =>
            prev.map((f) =>
              f.id === tempId
                ? {
                    ...f,
                    id: response.document_id,
                    status: 'success',
                    numChunks: response.num_chunks,
                  }
                : f
            )
          );

          uploadedDocIds.push(response.document_id);
        } catch (error) {
          // Update file status to error
          setUploadedFiles((prev) =>
            prev.map((f) =>
              f.id === tempId
                ? {
                    ...f,
                    status: 'error',
                    errorMessage: error instanceof Error ? error.message : 'Upload failed',
                  }
                : f
            )
          );
        }
      }

      setIsUploading(false);

      // Notify parent component
      if (onUploadComplete && uploadedDocIds.length > 0) {
        onUploadComplete(uploadedDocIds);
      }
    },
    [onUploadComplete, chunkingStrategy, chunkSize, chunkOverlap]
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'application/pdf': ['.pdf'],
      'text/plain': ['.txt'],
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ['.docx'],
    },
    maxFiles,
    disabled: isUploading,
  });

  const removeFile = (fileId: string) => {
    setUploadedFiles((prev) => prev.filter((f) => f.id !== fileId));
  };

  const formatFileSize = (bytes: number): string => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
  };

  // 목록 한 줄에 파일명이 길게 흐르지 않도록 자른다. 잘린 이름만으로는 어떤
  // 문서인지 알 수 없으므로, 전체 이름은 title로 남겨 hover에서 보이게 한다.
  const truncateFileName = (name: string): string =>
    name.length > MAX_FILE_NAME_LENGTH
      ? `${name.slice(0, MAX_FILE_NAME_LENGTH)}...`
      : name;

  return (
    <div className="file-uploader">
      <div className="upload-options">
        <button
          className="advanced-options-toggle"
          onClick={() => setShowAdvanced(!showAdvanced)}
          type="button"
        >
          <span>고급 설정 (청킹)</span>
          {showAdvanced ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
        </button>

        {showAdvanced && (
          <div className="advanced-options-panel">
            <label className="option-field">
              <span>청킹 전략</span>
              <select
                value={chunkingStrategy}
                onChange={(e) => setChunkingStrategy(e.target.value as ChunkingStrategy)}
                disabled={isUploading}
              >
                <option value="default">기본 (문자 수 기준)</option>
                <option value="semantic">시멘틱 (의미 단위)</option>
              </select>
            </label>

            {chunkingStrategy === 'default' && (
              <>
                <label className="option-field">
                  <span>Chunk Size</span>
                  <input
                    type="number"
                    min={1}
                    placeholder="기본값 사용"
                    value={chunkSize}
                    onChange={(e) => setChunkSize(e.target.value)}
                    disabled={isUploading}
                  />
                </label>
                <label className="option-field">
                  <span>Chunk Overlap</span>
                  <input
                    type="number"
                    min={0}
                    placeholder="기본값 사용"
                    value={chunkOverlap}
                    onChange={(e) => setChunkOverlap(e.target.value)}
                    disabled={isUploading}
                  />
                </label>
              </>
            )}
          </div>
        )}
      </div>

      <div
        {...getRootProps()}
        className={`dropzone ${isDragActive ? 'active' : ''} ${
          isUploading ? 'disabled' : ''
        }`}
      >
        <input {...getInputProps()} />
        <Upload className="upload-icon" size={48} />
        {isDragActive ? (
          <p>Drop the files here...</p>
        ) : (
          <>
            <p>Drag & drop files here, or click to select</p>
            <p className="file-types">Supports PDF, TXT, DOCX (max {maxFiles} files)</p>
          </>
        )}
      </div>

      {uploadedFiles.length > 0 && (
        <div className="uploaded-files-list">
          <h3>Uploaded Documents</h3>
          {uploadedFiles.map((file) => (
            <div key={file.id} className={`file-item ${file.status}`}>
              <div className="file-info">
                <FileText className="file-icon" size={20} />
                <div className="file-details">
                  <span className="file-name" title={file.name}>
                    {truncateFileName(file.name)}
                  </span>
                  <span className="file-meta">
                    {formatFileSize(file.size)}
                    {file.numChunks && ` • ${file.numChunks} chunks`}
                  </span>
                  {file.errorMessage && (
                    <span className="error-message">{file.errorMessage}</span>
                  )}
                </div>
              </div>

              <div className="file-status">
                {file.status === 'uploading' && (
                  <div className="spinner"></div>
                )}
                {file.status === 'success' && (
                  <CheckCircle className="status-icon success" size={20} />
                )}
                {file.status === 'error' && (
                  <AlertCircle className="status-icon error" size={20} />
                )}
                {file.status !== 'uploading' && (
                  <button
                    className="remove-button"
                    onClick={() => removeFile(file.id)}
                    aria-label="Remove file"
                  >
                    <X size={16} />
                  </button>
                )}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};
