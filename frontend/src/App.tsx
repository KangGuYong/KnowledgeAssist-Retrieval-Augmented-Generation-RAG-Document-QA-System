import { useState } from 'react';
import { FileUploader } from './components/FileUploader';
import { ChatWindow } from './components/ChatWindow';
import { ParsedDocumentViewer } from './components/ParsedDocumentViewer';
import './styles/App.css';

type ActiveTab = 'chat' | 'viewer';

function App() {
  const [uploadedDocumentIds, setUploadedDocumentIds] = useState<string[]>([]);
  const [activeTab, setActiveTab] = useState<ActiveTab>('chat');

  const handleUploadComplete = (documentIds: string[]) => {
    setUploadedDocumentIds((prev) => [...prev, ...documentIds]);
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
        </div>
        <div className="chat-section" hidden={activeTab !== 'chat'}>
          <ChatWindow documentIds={uploadedDocumentIds} />
        </div>
        <div className="app-main-full" hidden={activeTab !== 'viewer'}>
          <ParsedDocumentViewer />
        </div>
      </main>
    </div>
  );
}

export default App;
