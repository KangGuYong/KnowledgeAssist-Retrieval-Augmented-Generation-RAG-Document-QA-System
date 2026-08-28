import React from 'react';
import { FileText } from 'lucide-react';
import { SourceDocument } from '../types/api.types';
import '../styles/SourceCitation.css';

interface SourceCitationProps {
  source: SourceDocument;
  index: number;
}

export const SourceCitation: React.FC<SourceCitationProps> = ({ source, index }) => {
  return (
    <div className="source-citation">
      <div className="source-header">
        <FileText size={16} />
        <span className="source-index">[{index + 1}]</span>
        <span className="source-filename">{source.document_name}</span>
        {source.page !== undefined && (
          <span className="source-page">Page {source.page}</span>
        )}
      </div>

      <div className="source-content">
        <p>{source.content}</p>
      </div>

      {source.similarity_score !== undefined && (
        <div className="source-score">
          Relevance: {(source.similarity_score * 100).toFixed(1)}%
        </div>
      )}
    </div>
  );
};
