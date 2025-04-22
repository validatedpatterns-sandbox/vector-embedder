import logging
from typing import List, Optional

import redis
from langchain_community.vectorstores.redis import Redis as RedisVectorStore
from langchain_core.documents import Document

from vector_db.db_provider import DBProvider

logger = logging.getLogger(__name__)


class RedisProvider(DBProvider):
    """
    Redis-backed vector DB provider using RediSearch and LangChain's Redis integration.

    This provider connects to a Redis instance, checks if the specified index exists,
    and either loads from it or creates a new index on first insert. Vectors are stored
    using the RediSearch module with configurable schema.

    Attributes:
        redis_client (redis.Redis): Raw Redis client for low-level access.
        db (Optional[RedisVectorStore]): LangChain vector store, lazily created on first add.

    Args:
        embedding_model (str): Name of the embedding model to use for text chunks.
        url (str): Redis connection string (e.g., "redis://localhost:6379").
        index (str): RediSearch index name to use for vector storage.
        schema (str): Path to schema file where the RediSearch index definition is written.

    Example:
        >>> from vector_db.redis_provider import RedisProvider
        >>> provider = RedisProvider(
        ...     embedding_model="BAAI/bge-large-en-v1.5",
        ...     url="redis://localhost:6379",
        ...     index="validated_docs",
        ...     schema="redis_schema.yaml"
        ... )
        >>> provider.add_documents(docs)
    """

    def __init__(self, embedding_model: str, url: str, index: str, schema: str):
        """
        Initialize a Redis-backed vector store provider.

        Args:
            embedding_model (str): HuggingFace model for embeddings.
            url (str): Redis connection string.
            index (str): Name of the RediSearch index to use.
            schema (str): Path to write RediSearch schema YAML (used on creation).
        """
        super().__init__(embedding_model)
        self.url = url
        self.index = index
        self.schema = schema
        self.db: Optional[RedisVectorStore] = None

        try:
            self.redis_client = redis.from_url(self.url)
            self.redis_client.ping()
            logger.info("Connected to Redis instance at %s", self.url)
        except Exception:
            logger.exception("Failed to connect to Redis at %s", self.url)
            raise

        if self._index_exists():
            logger.info("Using existing Redis index: %s", self.index)
            self.db = RedisVectorStore.from_existing_index(
                embedding=self.embeddings,
                redis_url=self.url,
                index_name=self.index,
                schema=self.schema,
            )
        else:
            logger.info(
                "Redis index %s does not exist. Will create on first add_documents call.",
                self.index,
            )

    def _index_exists(self) -> bool:
        """
        Check whether the Redis index already exists.

        Returns:
            bool: True if the index exists, False otherwise.
        """
        try:
            self.redis_client.ft(self.index).info()
            return True
        except Exception:
            return False

    def add_documents(self, docs: List[Document]) -> None:
        """
        Add a list of documents to the Redis vector store.

        Args:
            docs (List[Document]): LangChain document chunks to embed and store.
        """
        if self.db is None:
            logger.info("Creating new Redis index: %s", self.index)
            self.db = RedisVectorStore.from_documents(
                documents=docs,
                embedding=self.embeddings,
                redis_url=self.url,
                index_name=self.index,
            )
            logger.info("Writing Redis schema to file: %s", self.schema)
            self.db.write_schema(self.schema)
        else:
            self.db.add_documents(docs)
