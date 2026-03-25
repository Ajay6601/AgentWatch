from functools import lru_cache

import torch
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Database
    database_url: str = "postgresql+asyncpg://agentwatch:agentwatch@localhost:5432/agentwatch"
    database_url_sync: str = "postgresql://agentwatch:agentwatch@localhost:5432/agentwatch"

    # LLM (for judge labeling)
    openai_api_key: str = ""
    judge_model: str = "gpt-4o-mini"
    judge_max_concurrent: int = 5

    # ML
    embedding_model: str = "all-MiniLM-L6-v2"
    setfit_base_model: str = "sentence-transformers/paraphrase-MiniLM-L6-v2"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    classifier_batch_size: int = 64

    # FAISS
    faiss_index_path: str = "./data/faiss_index"

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    model_config = {"env_file": ".env", "extra": "ignore"}


@lru_cache
def get_settings() -> Settings:
    return Settings()
