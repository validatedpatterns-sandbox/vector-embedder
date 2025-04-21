import json
import logging
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
    def _parse_log_level(log_level_name: str) -> int:
        log_levels = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL,
        }
        if log_level_name not in log_levels:
            raise ValueError(
                f"Invalid LOG_LEVEL: '{log_level_name}'. Must be one of: {', '.join(log_levels.keys())}"
            )
        return log_levels[log_level_name]

    @staticmethod
    def _init_db_provider(db_type: str) -> DBProvider:
        """
        Initialize the correct DBProvider subclass based on DB_TYPE.
        """
        get = Config._get_required_env_var
        db_type = db_type.upper()
        embedding_model = get("EMBEDDING_MODEL")

        if db_type == "REDIS":
            url = get("REDIS_URL")
            index = os.getenv("REDIS_INDEX", "docs")
            schema = os.getenv("REDIS_SCHEMA", "redis_schema.yaml")
            return RedisProvider(embedding_model, url, index, schema)

        elif db_type == "ELASTIC":
            url = get("ELASTIC_URL")
            password = get("ELASTIC_PASSWORD")
            index = os.getenv("ELASTIC_INDEX", "docs")
            user = os.getenv("ELASTIC_USER", "elastic")
            return ElasticProvider(embedding_model, url, password, index, user)

        elif db_type == "PGVECTOR":
            url = get("PGVECTOR_URL")
            collection = get("PGVECTOR_COLLECTION_NAME")
            return PGVectorProvider(embedding_model, url, collection)

        elif db_type == "SQLSERVER":
            host = get("SQLSERVER_HOST")
            port = get("SQLSERVER_PORT")
            user = get("SQLSERVER_USER")
            password = get("SQLSERVER_PASSWORD")
            database = get("SQLSERVER_DB")
            table = get("SQLSERVER_TABLE")
            driver = get("SQLSERVER_DRIVER")
            return SQLServerProvider(
                embedding_model, host, port, user, password, database, table, driver
            )

        elif db_type == "QDRANT":
            url = get("QDRANT_URL")
            collection = get("QDRANT_COLLECTION")
            return QdrantProvider(embedding_model, url, collection)

        elif db_type == "DRYRUN":
            return DryRunProvider(embedding_model)

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

        # Initialize logger
        log_level = get("LOG_LEVEL").upper()
        logging.basicConfig(level=Config._parse_log_level(log_level))
        logger = logging.getLogger(__name__)
        logger.debug("Logging initialized at level: %s", log_level)

        # Initialize db
        db_type = get("DB_TYPE")
        db_provider = Config._init_db_provider(db_type)

        # Web URLs
        try:
            web_sources = json.loads(get("WEB_SOURCES"))
        except json.JSONDecodeError as e:
            raise ValueError(f"WEB_SOURCES must be a valid JSON list: {e}")

        # Repo sources
        try:
            repo_sources = json.loads(get("REPO_SOURCES"))
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid REPO_SOURCES JSON: {e}") from e

        # Embedding settings
        chunk_size = int(get("CHUNK_SIZE"))
        chunk_overlap = int(get("CHUNK_OVERLAP"))

        # Misc
        temp_dir = get("TEMP_DIR")

        return Config(
            db_provider=db_provider,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            web_sources=web_sources,
            repo_sources=repo_sources,
            temp_dir=temp_dir,
            log_level=log_level,
        )
