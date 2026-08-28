from pydantic_settings import BaseSettings
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

    # Vector Store
    chroma_persist_dir: str = "app/storage/chroma_db"
    collection_name: str = "documents_kure_v1"

    # Embeddings
    embedding_model: str = "nlpai-lab/KURE-v1"

    # LLM Configuration
    llm_provider: str = "ollama"
    ollama_base_url: str = "http://192.168.0.169:11434"
    llm_model: str = "gemma4:26b-a4b-it-q4_K_M"
    llm_temperature: float = 0.0
    max_tokens: int = 2000

    # OCR (PDF image regions -> text)
    ocr_enabled: bool = True
    ocr_rec_model: str = "korean_PP-OCRv5_mobile_rec"
    ocr_det_model: str = "PP-OCRv5_mobile_det"
    ocr_device: str = "cpu"  # "cpu", "gpu", "gpu:0", ...
    ocr_min_score: float = 0.5  # Drop recognitions below this confidence
    ocr_use_textline_orientation: bool = False
    # PaddleOCR segfaults in a process that has torch loaded (paddlex imports
    # modelscope, which imports torch), so OCR runs in its own process.
    ocr_isolate_process: bool = True
    ocr_timeout: float = 120.0  # Per image, in the worker process
    ocr_dpi: int = 200  # Render resolution for image regions
    ocr_min_image_size: float = 40.0  # Skip icons/rules smaller than this (pt)
    ocr_max_images_per_page: int = 20  # Above this, OCR the whole page once
    ocr_page_text_threshold: int = 30  # Fewer native chars => treat as scanned
    ocr_row_tolerance: float = 10.0  # Blocks within this gap (pt) share a row
    ocr_layout_order: str = "position"  # "position" or "native" block order
    ocr_block_prefix: str = "[이미지 텍스트]"
    ocr_keep_empty_placeholder: bool = False
    ocr_empty_placeholder: str = "[이미지]"
    image_storage_dir: str = "app/storage/images"  # 도표 이미지 저장 경로

    # RAG Configuration
    chunk_size: int = 1000
    chunk_overlap: int = 200
    retrieval_k: int = 10  # Number of chunks to retrieve

    # Chunking strategy
    chunking_strategy: str = "default"  # "default" | "semantic"
    semantic_chunker_breakpoint_type: str = "percentile"  # percentile | standard_deviation | interquartile

    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Cached settings instance."""
    return Settings()
