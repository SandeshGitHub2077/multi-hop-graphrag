import os
import yaml
from pathlib import Path
from typing import Any, Optional


class Config:
    """Configuration loader - loads config.yaml with env var override capability.
    
    Env vars override config.yaml values:
      - NEO4J_URI
      - NEO4J_USER
      - NEO4J_PASSWORD
      - EMBEDDING_MODEL
      - LLM_MODEL
      - INDEX_DIR
      - DATA_DIR
    """

    _instance: Optional["Config"] = None
    _config: dict[str, Any] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load()
        return cls._instance

    def _load(self) -> None:
        config_path = Path(__file__).parent.parent / "config.yaml"
        
        if config_path.exists():
            with open(config_path) as f:
                self._config = yaml.safe_load(f) or {}
        else:
            self._config = {}

    @property
    def neo4j_uri(self) -> str:
        return os.environ.get("NEO4J_URI", self._config.get("graph", {}).get("uri", "bolt://localhost:7687"))

    @property
    def neo4j_user(self) -> str:
        return os.environ.get("NEO4J_USER", self._config.get("graph", {}).get("username", "neo4j"))

    @property
    def neo4j_password(self) -> str:
        return os.environ.get("NEO4J_PASSWORD", self._config.get("graph", {}).get("password", "password"))

    @property
    def embedding_model(self) -> str:
        return os.environ.get("EMBEDDING_MODEL", self._config.get("embeddings", {}).get("model", "BAAI/bge-base-en-v1.5"))

    @property
    def embedding_dimension(self) -> int:
        return self._config.get("embeddings", {}).get("dimension", 768)

    @property
    def llm_model(self) -> str:
        return os.environ.get("LLM_MODEL", "qwen3:8b")

    @property
    def index_dir(self) -> str:
        return os.environ.get("INDEX_DIR", self._config.get("vector_store", {}).get("index_dir", "index"))

    @property
    def data_dir(self) -> str:
        return os.environ.get("DATA_DIR", self._config.get("data", {}).get("data_dir", "Data"))

    def get(self, key: str, default: Any = None) -> Any:
        return self._config.get(key, default)


config = Config()