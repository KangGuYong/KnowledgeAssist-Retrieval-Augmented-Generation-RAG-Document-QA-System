import { useState } from 'react';
import { FileUploader } from './components/FileUploader';
import { ChatWindow } from './components/ChatWindow';
import './styles/App.css';

function App() {
  const [uploadedDocumentIds, setUploadedDocumentIds] = useState<string[]>([]);

  const handleUploadComplete = (documentIds: string[]) => {
    setUploadedDocumentIds((prev) => [...prev, ...documentIds]);
  };

  return (
    <div className="app">
      <header className="app-header">
        <h1>Knowledge Assist RAG</h1>
        <p>Upload documents and chat with them using AI</p>
      </header>

      <main className="app-main">
        <div className="sidebar">
          <FileUploader onUploadComplete={handleUploadComplete} />
        </div>

        <div className="chat-section">
          <ChatWindow documentIds={uploadedDocumentIds} />
        </div>
      </main>
    </div>
  );
}

export default App;
