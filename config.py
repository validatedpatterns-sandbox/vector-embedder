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
    Global configuration object for embedding and vector DB ingestion jobs.

    This class loads configuration from environment variables and initializes
    all the required components (e.g., DB providers, chunking strategy, input sources).

    Attributes:
        db_provider (DBProvider): Initialized provider for a vector database.
        chunk_size (int): Character length for each document chunk.
        chunk_overlap (int): Number of overlapping characters between adjacent chunks.
        web_sources (List[str]): List of web URLs to scrape and embed.
        repo_sources (List[Dict]): Repositories and glob patterns for file discovery.
        temp_dir (str): Path to a temporary working directory.
        log_level (int): Log verbosity level.

    Example:
        >>> config = Config.load()
        >>> print(config.chunk_size)
        >>> config.db_provider.add_documents(docs)
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
        """
        Retrieve a required environment variable or raise an error.

        Args:
            key (str): The environment variable name.

        Returns:
            str: The value of the environment variable.

        Raises:
            ValueError: If the variable is not defined.
        """
        value = os.getenv(key)
        if not value:
            raise ValueError(f"{key} environment variable is required.")
        return value

    @staticmethod
    def _parse_log_level(log_level_name: str) -> int:
        """
        Convert a string log level into a `logging` module constant.

        Args:
            log_level_name (str): One of DEBUG, INFO, WARNING, ERROR, CRITICAL.

        Returns:
            int: Corresponding `logging` level.

        Raises:
            ValueError: If an invalid level is provided.
        """
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
        Factory method to initialize the correct DB provider from environment variables.

        Args:
            db_type (str): Type of DB specified via `DB_TYPE` (e.g., REDIS, PGVECTOR, QDRANT, etc.)

        Returns:
            DBProvider: Initialized instance of a provider subclass.

        Raises:
            ValueError: If the DB type is unsupported or required vars are missing.
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
        Load application settings from `.env` variables into a typed config object.

        This includes logging level setup, DB provider initialization, and input
        source validation.

        Returns:
            Config: A fully-initialized configuration object.

        Raises:
            ValueError: If required environment variables are missing or malformed.
        """
        load_dotenv()
        get = Config._get_required_env_var

        # Logging setup
        log_level = get("LOG_LEVEL").upper()
        logging.basicConfig(level=Config._parse_log_level(log_level))
        logger = logging.getLogger(__name__)
        logger.debug("Logging initialized at level: %s", log_level)

        # Database backend
        db_type = get("DB_TYPE")
        db_provider = Config._init_db_provider(db_type)

        # Web source URLs
        try:
            web_sources = json.loads(get("WEB_SOURCES"))
        except json.JSONDecodeError as e:
            raise ValueError(f"WEB_SOURCES must be a valid JSON list: {e}")

        # Git repositories and file matchers
        try:
            repo_sources = json.loads(get("REPO_SOURCES"))
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid REPO_SOURCES JSON: {e}") from e

        # Embedding chunking strategy
        chunk_size = int(get("CHUNK_SIZE"))
        chunk_overlap = int(get("CHUNK_OVERLAP"))

        # Temporary file location
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
