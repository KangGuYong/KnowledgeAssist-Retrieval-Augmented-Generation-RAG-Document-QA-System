import React, { useState } from 'react';
import { FileText, Image as ImageIcon } from 'lucide-react';
import { SourceDocument } from '../types/api.types';
import '../styles/SourceCitation.css';

interface SourceCitationProps {
  source: SourceDocument;
  index: number;
}

export const SourceCitation: React.FC<SourceCitationProps> = ({ source, index }) => {
  const [lightboxUrl, setLightboxUrl] = useState<string | null>(null);
  const hasImages = source.image_urls.length > 0;

  return (
    <div className="source-citation">
      <div className="source-header">
        <FileText size={16} />
        <span className="source-index">[{index + 1}]</span>
        <span className="source-filename">{source.document_name}</span>
        {source.page !== undefined && (
          <span className="source-page">Page {source.page}</span>
        )}
        {hasImages && (
          <span className="source-badge">
            <ImageIcon size={12} /> 도표
          </span>
        )}
      </div>

      {hasImages && (
        <div className="source-images">
          {source.image_urls.map((url) => (
            <img
              key={url}
              src={url}
              alt="문서 도표"
              loading="lazy"
              className="source-image-thumb"
              onClick={() => setLightboxUrl(url)}
              onError={(e) => {
                (e.target as HTMLImageElement).style.display = 'none';
              }}
            />
          ))}
        </div>
      )}

      <div className="source-content">
        <p>{source.content}</p>
      </div>

      {source.similarity_score != null && (
        <div className="source-score">
          Relevance: {(source.similarity_score * 100).toFixed(1)}%
        </div>
      )}

      {lightboxUrl && (
        <div className="source-lightbox" onClick={() => setLightboxUrl(null)}>
          <img src={lightboxUrl} alt="문서 도표 확대" />
        </div>
      )}
    </div>
  );
};
