from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    app_name: str = Field(default="Insurance Policy Chatbot", alias="APP_NAME")
    app_env: str = Field(default="development", alias="APP_ENV")
    app_host: str = Field(default="0.0.0.0", alias="APP_HOST")
    app_port: int = Field(default=8000, alias="APP_PORT")
    policy_seed_url: str = Field(
        default=(
            "https://www.uhcprovider.com/en/policies-protocols/commercial-policies/"
            "commercial-medical-drug-policies.html"
        ),
        alias="POLICY_SEED_URL",
    )
    streamlit_backend_url: str = Field(
        default="http://localhost:8000",
        alias="STREAMLIT_BACKEND_URL",
    )
    admin_password: str = Field(default="", alias="ADMIN_PASSWORD")
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    openai_embedding_model: str = Field(
        default="text-embedding-3-small",
        alias="OPENAI_EMBEDDING_MODEL",
    )
    openai_chat_model: str = Field(
        default="gpt-4o-mini",
        alias="OPENAI_CHAT_MODEL",
    )
    qdrant_url: str = Field(default="", alias="QDRANT_URL")
    qdrant_api_key: str = Field(default="", alias="QDRANT_API_KEY")
    qdrant_collection_name: str = Field(
        default="insurance_policy_chunks",
        alias="QDRANT_COLLECTION_NAME",
    )
    qdrant_vector_size: int = Field(default=1536, alias="QDRANT_VECTOR_SIZE")
    retrieval_score_threshold: float = Field(default=0.75, alias="RETRIEVAL_SCORE_THRESHOLD")


@lru_cache
def get_settings() -> Settings:
    return Settings()
