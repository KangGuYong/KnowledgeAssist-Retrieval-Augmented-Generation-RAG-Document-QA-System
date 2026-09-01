import { useEffect, useState } from 'react';
import { FileUploader } from './components/FileUploader';
import { ChatWindow } from './components/ChatWindow';
import { ParsedDocumentViewer } from './components/ParsedDocumentViewer';
import { DocumentSelector } from './components/DocumentSelector';
import { apiService } from './services/api';
import { ParsedDocumentSummary } from './types/api.types';
import './styles/App.css';

type ActiveTab = 'chat' | 'viewer';

function App() {
  // 파싱된 문서 목록은 여기 한 곳에서만 관리한다 - 채팅 선택 목록과 문서
  // 뷰어가 각자 목록을 받아오면 업로드/삭제 후 서로 어긋나기 때문.
  const [documents, setDocuments] = useState<ParsedDocumentSummary[]>([]);
  const [documentsError, setDocumentsError] = useState<string | null>(null);
  const [selectedDocumentIds, setSelectedDocumentIds] = useState<string[]>([]);
  const [hasInitializedSelection, setHasInitializedSelection] = useState(false);
  const [activeTab, setActiveTab] = useState<ActiveTab>('chat');

  const refreshDocuments = () => {
    apiService
      .getParsedDocuments()
      .then((list) => {
        setDocuments(list);
        setDocumentsError(null);
        // 처음 목록을 불러올 때만 전체 선택으로 초기화한다 - 이후 재조회
        // (업로드/삭제)에서는 사용자가 직접 고른 선택을 덮어쓰지 않는다.
        if (!hasInitializedSelection) {
          setSelectedDocumentIds(list.map((doc) => doc.document_id));
          setHasInitializedSelection(true);
        }
      })
      .catch((err: Error) => {
        setDocumentsError(err.message);
        console.error('Failed to load documents:', err.message);
      });
  };

  useEffect(() => {
    refreshDocuments();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const handleUploadComplete = (documentIds: string[]) => {
    // 방금 올린 문서는 목록이 새로고침되기 전이라도 바로 선택된 상태로
    // 대화에 쓸 수 있어야 한다.
    setSelectedDocumentIds((prev) => Array.from(new Set([...prev, ...documentIds])));
    refreshDocuments();
  };

  const toggleDocumentSelected = (documentId: string) => {
    setSelectedDocumentIds((prev) =>
      prev.includes(documentId) ? prev.filter((id) => id !== documentId) : [...prev, documentId]
    );
  };

  const selectAllDocuments = () => setSelectedDocumentIds(documents.map((doc) => doc.document_id));
  const selectNoDocuments = () => setSelectedDocumentIds([]);

  const handleDocumentDeleted = (documentId: string) => {
    setDocuments((prev) => prev.filter((doc) => doc.document_id !== documentId));
    setSelectedDocumentIds((prev) => prev.filter((id) => id !== documentId));
  };

  return (
    <div className="app">
      <header className="app-header">
        <h1>Knowledge Assist RAG</h1>
        <p>Upload documents and chat with them using AI</p>
        <nav className="app-tabs">
          <button
            className={activeTab === 'chat' ? 'app-tab active' : 'app-tab'}
            onClick={() => setActiveTab('chat')}
          >
            채팅
          </button>
          <button
            className={activeTab === 'viewer' ? 'app-tab active' : 'app-tab'}
            onClick={() => setActiveTab('viewer')}
          >
            문서 뷰어
          </button>
        </nav>
      </header>

      <main className="app-main">
        <div className="sidebar" hidden={activeTab !== 'chat'}>
          <FileUploader onUploadComplete={handleUploadComplete} />
          <DocumentSelector
            documents={documents}
            selectedIds={selectedDocumentIds}
            onToggle={toggleDocumentSelected}
            onSelectAll={selectAllDocuments}
            onSelectNone={selectNoDocuments}
          />
        </div>
        <div className="chat-section" hidden={activeTab !== 'chat'}>
          <ChatWindow documentIds={selectedDocumentIds} />
        </div>
        <div className="app-main-full" hidden={activeTab !== 'viewer'}>
          <ParsedDocumentViewer
            documents={documents}
            loadError={documentsError}
            onDocumentDeleted={handleDocumentDeleted}
          />
        </div>
      </main>
    </div>
  );
}

export default App;
