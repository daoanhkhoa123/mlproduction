from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List

_ENV_DIR = Path(__file__).parent
_ENV_FILE_ENCODING = "utf-8"

class SecretKeys(BaseSettings):
    data_link: str  
    ngrok_token: str

    model_config = SettingsConfigDict(
        env_file=_ENV_DIR / ".key",
        env_file_encoding=_ENV_FILE_ENCODING,
        extra="ignore"
    )

class CloudSetting(BaseSettings):
    cloud_dependencies: List[str]
    
    model_config = SettingsConfigDict(
        env_file=_ENV_DIR / ".env",
        env_file_encoding=_ENV_FILE_ENCODING,
        extra="ignore"
    )