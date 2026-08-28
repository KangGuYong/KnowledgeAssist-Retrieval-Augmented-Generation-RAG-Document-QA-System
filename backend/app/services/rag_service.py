from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_models import ChatOllama
from typing import Optional, Dict
import logging
import uuid

from app.config import get_settings
from app.services.vector_store import get_vector_store
from app.api.models.responses import SourceDocument

logger = logging.getLogger(__name__)
settings = get_settings()


class RAGService:
    """Service for RAG-powered question answering."""

    def __init__(self):
        self.vector_store = get_vector_store()
        self.llm = self._initialize_llm()
        # Store conversation memories by conversation_id
        self.conversation_memories: Dict[str, ConversationBufferMemory] = {}

    def _initialize_llm(self):
        """Initialize the LLM based on provider setting."""
        if settings.llm_provider != "ollama":
            raise ValueError(f"Unsupported LLM provider: {settings.llm_provider}")

        return ChatOllama(
            base_url=settings.ollama_base_url,
            model=settings.llm_model,
            temperature=settings.llm_temperature,
        )

    def _get_or_create_memory(self, conversation_id: str) -> ConversationBufferMemory:
        """Get existing conversation memory or create new one."""
        if conversation_id not in self.conversation_memories:
            self.conversation_memories[conversation_id] = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                output_key="answer"
            )
        return self.conversation_memories[conversation_id]

    async def ask_question(
        self,
        question: str,
        conversation_id: Optional[str] = None,
        document_ids: Optional[list[str]] = None
    ) -> dict:
        """
        Answer a question using RAG.

        Args:
            question: User's question
            conversation_id: Optional conversation ID for context
            document_ids: Optional list of specific document IDs to search

        Returns:
            Dictionary with answer, sources, and metadata
        """
        # Generate conversation ID if not provided
        if not conversation_id:
            conversation_id = f"conv_{uuid.uuid4().hex[:12]}"

        # Get or create conversation memory
        memory = self._get_or_create_memory(conversation_id)

        # Set up retriever with optional document filtering
        search_kwargs = {"k": settings.retrieval_k}
        if document_ids:
            search_kwargs["filter"] = {"document_id": {"$in": document_ids}}

        retriever = self.vector_store.as_retriever(
            search_kwargs=search_kwargs
        )

        # Create conversational chain
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=retriever,
            memory=memory,
            return_source_documents=True,
            verbose=True
        )

        # Get response
        try:
            result = qa_chain({"question": question})

            # Format source documents
            sources = self._format_sources(result.get("source_documents", []))

            return {
                "answer": result["answer"],
                "sources": sources,
                "conversation_id": conversation_id,
                "message_id": f"msg_{uuid.uuid4().hex[:12]}"
            }

        except Exception as e:
            logger.error(f"Error in RAG pipeline: {e}")
            raise

    def _format_sources(self, source_docs: list) -> list[SourceDocument]:
        """Format source documents for response."""
        formatted_sources = []

        for doc in source_docs:
            document_id = doc.metadata.get("document_id", "")
            image_ids = doc.metadata.get("image_ids") or []
            image_urls = (
                [f"/api/v1/documents/{document_id}/images/{image_id}" for image_id in image_ids]
                if document_id
                else []
            )

            source = SourceDocument(
                content=doc.page_content,
                document_name=doc.metadata.get("filename", "Unknown"),
                document_id=document_id,
                page=doc.metadata.get("page"),
                chunk_index=doc.metadata.get("chunk_index", 0),
                similarity_score=None,  # Can add if using similarity_search_with_score
                image_urls=image_urls,
            )
            formatted_sources.append(source)

        return formatted_sources

    def clear_conversation(self, conversation_id: str) -> None:
        """Clear conversation history."""
        if conversation_id in self.conversation_memories:
            del self.conversation_memories[conversation_id]
            logger.info(f"Cleared conversation {conversation_id}")
