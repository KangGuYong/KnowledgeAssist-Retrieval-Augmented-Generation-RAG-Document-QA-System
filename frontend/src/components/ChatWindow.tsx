import React, { useState, useRef, useEffect } from 'react';
import { Send, Trash2 } from 'lucide-react';
import { apiService } from '../services/api';
import { Message as MessageType } from '../types/chat.types';
import { ChatRequest } from '../types/api.types';
import { Message } from './Message';
import '../styles/ChatWindow.css';

interface ChatWindowProps {
  documentIds?: string[];
}

export const ChatWindow: React.FC<ChatWindowProps> = ({ documentIds }) => {
  const [messages, setMessages] = useState<MessageType[]>([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [conversationId, setConversationId] = useState<string | undefined>(undefined);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSendMessage = async () => {
    if (!inputValue.trim() || isLoading) return;

    const userMessage: MessageType = {
      id: `user-${Date.now()}`,
      role: 'user',
      content: inputValue.trim(),
      timestamp: new Date(),
    };

    // Add user message to chat
    setMessages((prev) => [...prev, userMessage]);
    setInputValue('');
    setIsLoading(true);

    // Add temporary loading message
    const loadingMessage: MessageType = {
      id: `loading-${Date.now()}`,
      role: 'assistant',
      content: '',
      timestamp: new Date(),
      isLoading: true,
    };
    setMessages((prev) => [...prev, loadingMessage]);

    try {
      // Prepare request
      const request: ChatRequest = {
        question: userMessage.content,
        conversation_id: conversationId,
        document_ids: documentIds,
      };

      // Call API
      const response = await apiService.sendMessage(request);

      // Update conversation ID if this is the first message
      if (!conversationId) {
        setConversationId(response.conversation_id);
      }

      // Create assistant message
      const assistantMessage: MessageType = {
        id: response.message_id,
        role: 'assistant',
        content: response.answer,
        sources: response.sources,
        timestamp: new Date(response.timestamp),
        tokenUsage: response.token_usage,
      };

      // Replace loading message with actual response
      setMessages((prev) =>
        prev.map((msg) =>
          msg.id === loadingMessage.id ? assistantMessage : msg
        )
      );
    } catch (error) {
      // Remove loading message and show error
      setMessages((prev) => prev.filter((msg) => msg.id !== loadingMessage.id));

      const errorMessage: MessageType = {
        id: `error-${Date.now()}`,
        role: 'assistant',
        content: `Sorry, I encountered an error: ${
          error instanceof Error ? error.message : 'Unknown error'
        }`,
        timestamp: new Date(),
      };

      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
      inputRef.current?.focus();
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const handleClearChat = async () => {
    if (conversationId) {
      try {
        await apiService.clearConversation(conversationId);
      } catch (error) {
        console.error('Error clearing conversation:', error);
      }
    }
    setMessages([]);
    setConversationId(undefined);
  };

  return (
    <div className="chat-window">
      <div className="chat-header">
        <h2>Chat with Your Documents</h2>
        {messages.length > 0 && (
          <button
            className="clear-button"
            onClick={handleClearChat}
            disabled={isLoading}
            title="Clear chat"
          >
            <Trash2 size={18} />
          </button>
        )}
      </div>

      <div className="messages-container">
        {messages.length === 0 ? (
          <div className="empty-state">
            <p>Upload documents and start asking questions!</p>
            <p className="hint">
              Try asking: "What are the main topics in this document?" or "Summarize
              the key points"
            </p>
          </div>
        ) : (
          messages.map((message) => <Message key={message.id} message={message} />)
        )}
        <div ref={messagesEndRef} />
      </div>

      <div className="input-container">
        <textarea
          ref={inputRef}
          className="chat-input"
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask a question about your documents..."
          rows={1}
          disabled={isLoading}
        />
        <button
          className="send-button"
          onClick={handleSendMessage}
          disabled={!inputValue.trim() || isLoading}
          aria-label="Send message"
        >
          <Send size={20} />
        </button>
      </div>
    </div>
  );
};
