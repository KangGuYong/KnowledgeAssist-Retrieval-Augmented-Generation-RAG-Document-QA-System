import React, { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { Upload, FileText, X, CheckCircle, AlertCircle } from 'lucide-react';
import { apiService } from '../services/api';
import { UploadedFile } from '../types/chat.types';
import '../styles/FileUploader.css';

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

  const onDrop = useCallback(
    async (acceptedFiles: File[]) => {
      if (acceptedFiles.length === 0) return;

      setIsUploading(true);

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
          const response = await apiService.uploadFile(file);

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
    [onUploadComplete]
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

  return (
    <div className="file-uploader">
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
                  <span className="file-name">{file.name}</span>
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
