import json
import os
from dataclasses import dataclass
from typing import Dict, List

from dotenv import load_dotenv

from vector_db.db_provider import DBProvider
from vector_db.dryrun_provider import DryRunProvider
from vector_db.elastic_provider import ElasticProvider
from vector_db.pgvector_provider import PGVectorProvider
from vector_db.qdrant_provider import QdrantProvider
from vector_db.redis_provider import RedisProvider
from vector_db.sqlserver_provider import SQLServerProvider


@dataclass
class Config:
    """
    Application configuration loaded from environment variables.

    This centralizes all configuration values needed for the embedding job,
    including database provider setup, chunking behavior, document sources,
    and logging configuration.

    Use `Config.load()` to load and validate values from the current environment.
    """

    db_provider: DBProvider
    chunk_size: int
    chunk_overlap: int
    web_sources: List[str]
    repo_sources: List[Dict]
    temp_dir: str
    log_level: int

    @staticmethod
    def _get_required_env_var(key: str) -> str:
        value = os.getenv(key)
        if not value:
            raise ValueError(f"{key} environment variable is required.")
        return value

    @staticmethod
    def _init_db_provider(db_type: str) -> DBProvider:
        """
        Initialize the correct DBProvider subclass based on DB_TYPE.
        """
        db_type = db_type.upper()

        if db_type == "REDIS":
            url = Config._get_required_env_var("REDIS_URL")
            index = os.getenv("REDIS_INDEX", "docs")
            schema = os.getenv("REDIS_SCHEMA", "redis_schema.yaml")
            return RedisProvider(url, index, schema)

        elif db_type == "ELASTIC":
            url = Config._get_required_env_var("ELASTIC_URL")
            password = Config._get_required_env_var("ELASTIC_PASSWORD")
            index = os.getenv("ELASTIC_INDEX", "docs")
            user = os.getenv("ELASTIC_USER", "elastic")
            return ElasticProvider(url, password, index, user)

        elif db_type == "PGVECTOR":
            url = Config._get_required_env_var("PGVECTOR_URL")
            collection = Config._get_required_env_var("PGVECTOR_COLLECTION_NAME")
            return PGVectorProvider(url, collection)

        elif db_type == "SQLSERVER":
            return SQLServerProvider()  # Handles its own env var loading

        elif db_type == "QDRANT":
            url = Config._get_required_env_var("QDRANT_URL")
            collection = Config._get_required_env_var("QDRANT_COLLECTION")
            return QdrantProvider(url, collection)

        elif db_type == "DRYRUN":
            return DryRunProvider()

        raise ValueError(f"Unsupported DB_TYPE '{db_type}'")

    @staticmethod
    def load() -> "Config":
        """
        Load configuration from environment variables.

        All values are expected to be present in the environment and are validated.
        This method is the single point of truth for all configurable values used
        throughout the embedding pipeline.

        Returns:
            Config: A fully populated Config object with validated values.

        Raises:
            ValueError: If any required variable is missing or invalid.
        """
        load_dotenv()
        get = Config._get_required_env_var

        db_type = get("DB_TYPE")
        db_provider = Config._init_db_provider(db_type)

        chunk_size = int(get("CHUNK_SIZE"))
        chunk_overlap = int(get("CHUNK_OVERLAP"))
        temp_dir = get("TEMP_DIR")

        # Web URLs
        web_sources_raw = get("WEB_SOURCES")
        try:
            web_sources = json.loads(web_sources_raw)
        except json.JSONDecodeError as e:
            raise ValueError(f"WEB_SOURCES must be a valid JSON list: {e}")

        # Repo sources
        repo_sources_json = get("REPO_SOURCES")
        try:
            repo_sources = json.loads(repo_sources_json)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid REPO_SOURCES JSON: {e}") from e

        # Logging
        log_level_name = get("LOG_LEVEL").lower()
        log_levels = {
            "debug": 10,
            "info": 20,
            "warning": 30,
            "error": 40,
            "critical": 50,
        }
        if log_level_name not in log_levels:
            raise ValueError(
                f"Invalid LOG_LEVEL: '{log_level_name}'. Must be one of: {', '.join(log_levels)}"
            )
        log_level = log_levels[log_level_name]

        return Config(
            db_provider=db_provider,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            web_sources=web_sources,
            repo_sources=repo_sources,
            temp_dir=temp_dir,
            log_level=log_level,
        )
