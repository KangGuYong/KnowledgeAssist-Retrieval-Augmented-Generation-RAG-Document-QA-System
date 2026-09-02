from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # API Configuration
    app_name: str = "Knowledge Assist RAG API"
    app_version: str = "1.0.0"
    api_prefix: str = "/api/v1"
    debug: bool = False

    # CORS
    allowed_origins: list[str] = [
        "http://localhost:5173",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
    ]

    # File Upload
    max_upload_size: int = 10 * 1024 * 1024  # 10MB
    allowed_extensions: set[str] = {".pdf", ".txt", ".docx"}
    upload_dir: str = "app/storage/uploads"
    max_documents: int = 10  # Total documents allowed in the system at once

    # Vector Store
    chroma_persist_dir: str = "app/storage/chroma_db"
    collection_name: str = "documents_kure_v1"

    # Embeddings
    embedding_model: str = "nlpai-lab/KURE-v1"
    embedding_device: str = "cuda"  # "cpu", "cuda", "cuda:0", ...

    # LLM Configuration
    llm_provider: str = "ollama"
    ollama_base_url: str = "http://192.168.0.169:11434"
    llm_model: str = "gemma4:26b-a4b-it-q4_K_M"
    llm_temperature: float = 0.0
    max_tokens: int = 2000
    # 원격 Ollama가 잠깐 끊기거나 응답이 늦을 때만 다시 시도한다. 모델 없음·잘못된
    # 요청 같은 영구 실패는 몇 번을 걸어도 같으므로 재시도 대상이 아니다.
    # 1로 두면 재시도하지 않는다.
    llm_max_attempts: int = 3

    # OCR (PDF image regions -> text)
    ocr_enabled: bool = True  # Whether image blocks get PaddleOCR augmentation (MinerU parsing itself is controlled by mineru_enabled)
    ocr_rec_model: str = "korean_PP-OCRv5_mobile_rec"
    ocr_det_model: str = "PP-OCRv5_mobile_det"
    ocr_device: str = "cpu"  # "cpu", "gpu", "gpu:0", ...
    ocr_min_score: float = 0.5  # Drop recognitions below this confidence
    ocr_use_textline_orientation: bool = False
    # PaddleOCR segfaults in a process that has torch loaded (paddlex imports
    # modelscope, which imports torch), so OCR runs in its own process.
    ocr_isolate_process: bool = True
    ocr_timeout: float = 120.0  # Per image, in the worker process
    ocr_block_prefix: str = "[이미지 텍스트]"
    ocr_keep_empty_placeholder: bool = False
    ocr_empty_placeholder: str = "[이미지]"
    image_storage_dir: str = "app/storage/images"  # 도표 이미지 저장 경로

    # MinerU (PDF layout/table/formula parsing service)
    mineru_base_url: str = "http://127.0.0.1:8100"
    mineru_timeout: float = 300.0  # seconds; large PDFs take much longer than a single OCR call
    mineru_enabled: bool = True
    mineru_lang_list: list[str] = ["korean"]  # passed verbatim as /file_parse's lang_list form field

    # Parsed-result viewer: raw MinerU content_list blocks, saved alongside images
    parsed_storage_dir: str = "app/storage/parsed"

    # RAG Configuration
    chunk_size: int = 1000
    chunk_overlap: int = 200
    retrieval_k: int = 10  # Number of chunks to retrieve
    # 검색 결과를 관련성 순서 그대로 넣지 않고, 관련성 높은 청크를 컨텍스트
    # 양 끝으로 보낸다(Lost in the Middle 완화). 끄면 관련성 내림차순 그대로.
    retrieval_reorder: bool = True

    # Chunking strategy
    chunking_strategy: str = "default"  # "default" | "semantic"
    # Distances above this percentile of all consecutive-sentence distances
    # are treated as chunk breakpoints (5 Levels of Text Splitting, Level 4).
    semantic_chunker_breakpoint_percentile: float = 95.0
    # A breakpoint that would emit a chunk shorter than this is skipped, so a
    # run of short lines cannot produce chunks too small to answer anything.
    semantic_chunker_min_chunk_chars: int = 200

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False)


@lru_cache()
def get_settings() -> Settings:
    """Cached settings instance."""
    return Settings()
