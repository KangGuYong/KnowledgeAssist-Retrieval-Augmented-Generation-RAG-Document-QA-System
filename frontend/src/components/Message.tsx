import React, { useState } from 'react';
import { User, Bot, ChevronDown, ChevronUp } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import { Message as MessageType } from '../types/chat.types';
import { SourceCitation } from './SourceCitation';
import '../styles/Message.css';

interface MessageProps {
  message: MessageType;
}

export const Message: React.FC<MessageProps> = ({ message }) => {
  const [showSources, setShowSources] = useState(false);
  const isUser = message.role === 'user';

  return (
    <div className={`message ${message.role}`}>
      <div className="message-avatar">
        {isUser ? <User size={24} /> : <Bot size={24} />}
      </div>

      <div className="message-content">
        <div className="message-header">
          <span className="message-role">{isUser ? 'You' : 'Assistant'}</span>
          <span className="message-timestamp">
            {message.timestamp.toLocaleTimeString([], {
              hour: '2-digit',
              minute: '2-digit',
            })}
          </span>
        </div>

        <div className="message-text">
          {message.isLoading ? (
            <div className="typing-indicator">
              <span></span>
              <span></span>
              <span></span>
            </div>
          ) : (
            <ReactMarkdown>{message.content}</ReactMarkdown>
          )}
        </div>

        {!isUser && message.sources && message.sources.length > 0 && (
          <div className="message-sources">
            <button
              className="sources-toggle"
              onClick={() => setShowSources(!showSources)}
            >
              <span>
                {message.sources.length} source{message.sources.length !== 1 ? 's' : ''}
              </span>
              {showSources ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
            </button>

            {showSources && (
              <div className="sources-list">
                {message.sources.map((source, index) => (
                  <SourceCitation key={index} source={source} index={index} />
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};
