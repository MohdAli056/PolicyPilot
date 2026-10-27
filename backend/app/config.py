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

    # Document storage configuration
    documents_dir: str = "data/documents"
    metadata_dir: str = "data/metadata"
    max_upload_size_bytes: int = 10 * 1024 * 1024  # 10 MB default

    def get_documents_path(self):
        """Return resolved pathlib.Path for document storage."""
        from pathlib import Path

        p = Path(self.documents_dir)
        if not p.is_absolute():
            backend_root = Path(__file__).resolve().parent.parent
            p = backend_root / p
        return p.resolve()

    def get_metadata_path(self):
        """Return resolved pathlib.Path for metadata storage."""
        from pathlib import Path

        p = Path(self.metadata_dir)
        if not p.is_absolute():
            backend_root = Path(__file__).resolve().parent.parent
            p = backend_root / p
        return p.resolve()


settings = Settings()
