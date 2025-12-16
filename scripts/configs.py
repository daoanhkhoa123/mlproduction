from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

_ENV_FILE = Path(__file__).parent.parent / ".key"
_ENV_FILE_ENCODING = "utf-8"

class LLMApiKeys(BaseSettings):
    google_api_key: str  
    cerebras_api_key: str
    jina_api_key: str

    model_config = SettingsConfigDict(
        env_file=_ENV_FILE,
        env_file_encoding=_ENV_FILE_ENCODING,
        extra="ignore"
    )