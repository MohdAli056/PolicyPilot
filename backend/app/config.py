"""Application configuration loaded from environment variables."""

from pydantic import ConfigDict
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Central configuration for the PolicyPilot backend.

    Values are loaded from environment variables or a .env file.
    """

    model_config = ConfigDict(env_file=".env", env_file_encoding="utf-8")

    app_name: str = "PolicyPilot"
    app_version: str = "0.1.0"
    debug: bool = False

    # CORS: comma-separated list of allowed origins
    cors_origins: str = "http://localhost:3000"


settings = Settings()
