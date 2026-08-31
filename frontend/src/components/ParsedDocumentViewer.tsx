import React, { useEffect, useState } from 'react';
import DOMPurify from 'dompurify';
import { FileText } from 'lucide-react';
import { apiService } from '../services/api';
import { ParsedBlock, ParsedDocumentDetail, ParsedDocumentSummary } from '../types/api.types';
import '../styles/ParsedDocumentViewer.css';

function imageUrl(documentId: string, imageId: string): string {
  return `${import.meta.env.VITE_API_BASE_URL ?? ''}/api/v1/documents/${documentId}/images/${imageId}`;
}

function ParsedBlockView({ documentId, block }: { documentId: string; block: ParsedBlock }) {
  // A table block can carry both table_body (MinerU's HTML extraction) and
  // image_id (the screenshot MinerU actually parsed) at once - render both
  // so a user can compare them, rather than returning early on one.
  if (block.table_body) {
    return (
      <>
        <div
          className="parsed-block parsed-block-table"
          dangerouslySetInnerHTML={{ __html: DOMPurify.sanitize(block.table_body) }}
        />
        {block.image_id && (
          <img
            className="parsed-block parsed-block-image parsed-block-table-image"
            src={imageUrl(documentId, block.image_id)}
            alt="table 블록 스크린샷"
            loading="lazy"
          />
        )}
      </>
    );
  }

  if (block.type === 'equation' && block.text) {
    return (
      <pre className="parsed-block parsed-block-equation">
        <code>{block.text}</code>
      </pre>
    );
  }

  if (block.image_id) {
    return (
      <img
        className="parsed-block parsed-block-image"
        src={imageUrl(documentId, block.image_id)}
        alt={`${block.type} 블록`}
        loading="lazy"
      />
    );
  }

  if (block.text) {
    const isTitle = block.type === 'title';
    return (
      <p className={isTitle ? 'parsed-block parsed-block-title' : 'parsed-block parsed-block-text'}>
        {block.text}
      </p>
    );
  }

  return null;
}

export const ParsedDocumentViewer: React.FC = () => {
  const [documents, setDocuments] = useState<ParsedDocumentSummary[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [detail, setDetail] = useState<ParsedDocumentDetail | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    apiService
      .getParsedDocuments()
      .then(setDocuments)
      .catch((err: Error) => setError(err.message));
  }, []);

  useEffect(() => {
    if (!selectedId) {
      setDetail(null);
      return;
    }
    setLoading(true);
    setError(null);
    apiService
      .getParsedDocument(selectedId)
      .then(setDetail)
      .catch((err: Error) => setError(err.message))
      .finally(() => setLoading(false));
  }, [selectedId]);

  return (
    <div className="parsed-viewer">
      <div className="parsed-viewer-sidebar">
        <h2>파싱된 문서</h2>
        {documents.length === 0 && <p className="parsed-viewer-empty">아직 파싱된 문서가 없습니다.</p>}
        <ul>
          {documents.map((doc) => (
            <li key={doc.document_id}>
              <button
                className={doc.document_id === selectedId ? 'parsed-doc-item active' : 'parsed-doc-item'}
                onClick={() => setSelectedId(doc.document_id)}
              >
                <FileText size={14} />
                <span className="parsed-doc-filename">{doc.filename}</span>
                <span className="parsed-doc-pages">{doc.page_count}p</span>
              </button>
            </li>
          ))}
        </ul>
      </div>

      <div className="parsed-viewer-content">
        {error && <p className="parsed-viewer-error">{error}</p>}
        {loading && <p>불러오는 중...</p>}
        {!loading && !error && !detail && <p className="parsed-viewer-empty">왼쪽에서 문서를 선택하세요.</p>}
        {detail &&
          detail.pages.map((page) => (
            <section key={page.page_number} className="parsed-page">
              <h3>Page {page.page_number}</h3>
              {page.blocks.map((block, idx) => (
                <ParsedBlockView key={idx} documentId={detail.document_id} block={block} />
              ))}
            </section>
          ))}
      </div>
    </div>
  );
};
