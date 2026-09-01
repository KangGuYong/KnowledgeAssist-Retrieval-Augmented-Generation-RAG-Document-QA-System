import React, { useEffect, useState } from 'react';
import DOMPurify from 'dompurify';
import { FileText, Trash2 } from 'lucide-react';
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

interface ParsedDocumentViewerProps {
  /** 문서 삭제가 성공했을 때 알려준다 - App.tsx의 채팅용 문서 선택 목록도
   * 같이 갱신되어야 하기 때문 (그렇지 않으면 이미 삭제된 문서가 선택
   * 목록에 그대로 남는다). */
  onDocumentDeleted?: (documentId: string) => void;
}

export const ParsedDocumentViewer: React.FC<ParsedDocumentViewerProps> = ({ onDocumentDeleted }) => {
  const [documents, setDocuments] = useState<ParsedDocumentSummary[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [detail, setDetail] = useState<ParsedDocumentDetail | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [deletingId, setDeletingId] = useState<string | null>(null);

  useEffect(() => {
    apiService
      .getParsedDocuments()
      .then(setDocuments)
      .catch((err: Error) => setError(err.message));
  }, []);

  const handleDelete = async (e: React.MouseEvent, documentId: string, filename: string) => {
    e.stopPropagation();
    if (!window.confirm(`"${filename}" 문서를 삭제할까요? 되돌릴 수 없습니다.`)) return;

    setDeletingId(documentId);
    try {
      await apiService.deleteDocument(documentId);
      setDocuments((prev) => prev.filter((doc) => doc.document_id !== documentId));
      if (selectedId === documentId) {
        setSelectedId(null);
      }
      onDocumentDeleted?.(documentId);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setDeletingId(null);
    }
  };

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
            <li
              key={doc.document_id}
              className={doc.document_id === selectedId ? 'parsed-doc-item active' : 'parsed-doc-item'}
            >
              <button className="parsed-doc-select" onClick={() => setSelectedId(doc.document_id)}>
                <FileText size={14} />
                <span className="parsed-doc-filename">{doc.filename}</span>
                <span className="parsed-doc-pages">{doc.page_count}p</span>
              </button>
              <button
                className="parsed-doc-delete"
                onClick={(e) => handleDelete(e, doc.document_id, doc.filename)}
                disabled={deletingId === doc.document_id}
                title="문서 삭제"
              >
                <Trash2 size={14} />
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
