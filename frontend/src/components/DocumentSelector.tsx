import React from 'react';
import { FileText } from 'lucide-react';
import { ParsedDocumentSummary } from '../types/api.types';
import '../styles/DocumentSelector.css';

interface DocumentSelectorProps {
  documents: ParsedDocumentSummary[];
  selectedIds: string[];
  onToggle: (documentId: string) => void;
  onSelectAll: () => void;
  onSelectNone: () => void;
}

export const DocumentSelector: React.FC<DocumentSelectorProps> = ({
  documents,
  selectedIds,
  onToggle,
  onSelectAll,
  onSelectNone,
}) => {
  if (documents.length === 0) {
    return null;
  }

  return (
    <div className="document-selector">
      <div className="document-selector-header">
        <h3>대화에 사용할 문서</h3>
        <div className="document-selector-actions">
          <button type="button" onClick={onSelectAll}>
            전체 선택
          </button>
          <button type="button" onClick={onSelectNone}>
            전체 해제
          </button>
        </div>
      </div>
      <ul>
        {documents.map((doc) => (
          <li key={doc.document_id}>
            <label className="document-selector-item">
              <input
                type="checkbox"
                checked={selectedIds.includes(doc.document_id)}
                onChange={() => onToggle(doc.document_id)}
              />
              <FileText size={14} />
              <span className="document-selector-filename">{doc.filename}</span>
            </label>
          </li>
        ))}
      </ul>
    </div>
  );
};
