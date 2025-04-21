import logging
from typing import List, Optional

import redis
from langchain_community.vectorstores.redis import Redis as RedisVectorStore
from langchain_core.documents import Document

from vector_db.db_provider import DBProvider

logger = logging.getLogger(__name__)


class RedisProvider(DBProvider):
    """
    Redis-based vector DB provider using RediSearch and LangChain's Redis integration.

    Args:
        embedding_model (str): Embedding model to use
        url (str): Redis connection string (e.g. redis://localhost:6379)
        index (str): RediSearch index name (must be provided via .env)
        schema (str): Path to RediSearch schema YAML file (must be provided via .env)

    This provider will either load from an existing Redis index or defer creation
    until documents are available.

    Example:
        >>> provider = RedisProvider(
        ...     embedding_model="sentence-transformers/all-mpnet-base-v2",
        ...     url="redis://localhost:6379",
        ...     index="docs",
        ...     schema="redis_schema.yaml"
        ... )
        >>> provider.add_documents(chunks)
    """

    def __init__(self, embedding_model: str, url: str, index: str, schema: str):
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
        try:
            self.redis_client.ft(self.index).info()
            return True
        except Exception:
            return False

    def add_documents(self, docs: List[Document]) -> None:
        """
        Add document chunks to Redis vector store.

        Args:
            docs (List[Document]): Chunked LangChain documents to store.
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
