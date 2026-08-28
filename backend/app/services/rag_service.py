from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama
from typing import Optional, Dict
import logging
import uuid

from app.config import get_settings
from app.services.vector_store import get_vector_store
from app.api.models.responses import SourceDocument

logger = logging.getLogger(__name__)
settings = get_settings()

# 문맥 중 '[이미지 텍스트]'는 OCR로 추출된 것이라 오탈자가 있을 수 있다. 청크
# 본문에 이미 인라인으로 박혀 있으므로(pdf_ocr._format_ocr_block), 여기서는 LLM에게
# 그 마커를 어떻게 다뤄야 하는지만 알려준다.
QA_PROMPT = PromptTemplate(
    template="""다음 문맥을 참고해 질문에 답하라.
문맥 중 '[이미지 텍스트]'로 표시된 부분은 문서의 도표·이미지에서 문자 인식(OCR)으로
추출한 것이라 오탈자가 있을 수 있다. 이를 근거로 답할 때는 인명·지명 등 고유명사를
단정하지 말고, 해당 페이지의 도표를 직접 확인하도록 안내하라.

문맥:
{context}

질문: {question}
답변:""",
    input_variables=["context", "question"],
)


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
            combine_docs_chain_kwargs={"prompt": QA_PROMPT},
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
