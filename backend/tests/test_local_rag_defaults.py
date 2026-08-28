from app.config import Settings
from app.services.rag_service import RAGService


def test_defaults_target_kure_and_local_ollama():
    """A fresh installation must use the requested local RAG stack."""
    settings = Settings()

    assert settings.embedding_model == "nlpai-lab/KURE-v1"
    assert settings.collection_name == "documents_kure_v1"
    assert settings.llm_provider == "ollama"
    assert settings.ollama_base_url == "http://192.168.0.169:11434"
    assert settings.llm_model == "gemma4:26b-a4b-it-q4_K_M"


def test_rag_service_builds_an_ollama_chat_model():
    """The LLM adapter must connect to the configured local Ollama host."""
    service = RAGService.__new__(RAGService)
    llm = service._initialize_llm()

    assert llm.model == "gemma4:26b-a4b-it-q4_K_M"
    assert llm.base_url == "http://192.168.0.169:11434"
