"""Redis vector database provider implementation."""

import logging
from typing import List

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_redis import RedisVectorStore

from vector_db.db_provider import DBProvider

logger = logging.getLogger(__name__)


class RedisProvider(DBProvider):
    """
    Redis-backed vector DB provider using RediSearch and LangChain's Redis integration.

    This implementation uses Redis as a backend for storing vector embeddings, via the
    LangChain RedisVectorStore.

    Attributes:
        db (RedisVectorStore): LangChain-compatible Redis vector store instance.

    Args:
        embeddings (HuggingFaceEmbeddings): An initialized HuggingFace embeddings instance.
        url (str): Redis connection string (e.g., "redis://localhost:6379").
        index (str): RediSearch index name to use for vector storage.

    Example:
        >>> from langchain_huggingface import HuggingFaceEmbeddings
        >>> from vector_db.redis_provider import RedisProvider
        >>> embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")
        >>> provider = RedisProvider(
        ...     embeddings=embeddings,
        ...     url="redis://localhost:6379",
        ...     index="validated_docs"
        ... )
        >>> provider.add_documents(docs)
    """

    def __init__(self, embeddings: HuggingFaceEmbeddings, url: str, index: str):
        """
        Initialize a Redis-backed vector store provider.

        Args:
            embeddings (HuggingFaceEmbeddings): HuggingFace embeddings instance.
            url (str): Redis connection string.
            index (str): Name of the RediSearch index to use.
        """
        super().__init__(embeddings)

        self.db = RedisVectorStore(
            index_name=index, embeddings=self.embeddings, redis_url=url
        )

        logger.info("Connected to Redis at %s (index: %s)", url, index)

    def add_documents(self, docs: List[Document]) -> None:
        """
        Add a list of documents to the Redis vector store.

        Args:
            docs (List[Document]): LangChain Document objects to embed and store.
        """
        self.db.add_documents(docs)
