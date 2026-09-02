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


def test_qa_prompt_warns_about_ocr_marker():
    from app.services.rag_service import QA_PROMPT

    assert "[이미지 텍스트]" in QA_PROMPT.template
    assert "{context}" in QA_PROMPT.template
    assert "{question}" in QA_PROMPT.template


def test_retrieval_reorder_is_enabled_by_default():
    """Lost in the Middle 완화는 기본으로 켜져 있고, 끌 수 있어야 한다."""
    settings = Settings(_env_file=None)

    assert settings.retrieval_reorder is True


def test_qa_prompt_does_not_claim_the_context_is_relevance_ordered():
    """재배치 후에는 컨텍스트가 관련성 순이 아니므로 그렇게 말하면 안 된다."""
    from app.services.rag_service import QA_PROMPT

    assert "관련성 순" not in QA_PROMPT.template
