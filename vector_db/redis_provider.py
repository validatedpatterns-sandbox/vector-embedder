import logging
from typing import List

from langchain_core.documents import Document
from langchain_redis import RedisVectorStore

from vector_db.db_provider import DBProvider

logger = logging.getLogger(__name__)


class RedisProvider(DBProvider):
    """
    Redis-backed vector DB provider using RediSearch and LangChain's Redis integration.

    Attributes:
        db (RedisVectorStore): LangChain vector store

    Args:
        embedding_model (str): Name of the embedding model to use for text chunks.
        url (str): Redis connection string (e.g., "redis://localhost:6379").
        index (str): RediSearch index name to use for vector storage.

    Example:
        >>> from vector_db.redis_provider import RedisProvider
        >>> provider = RedisProvider(
        ...     embedding_model="BAAI/bge-large-en-v1.5",
        ...     url="redis://localhost:6379",
        ...     index="validated_docs"
        ... )
        >>> provider.add_documents(docs)
    """

    def __init__(self, embedding_model: str, url: str, index: str):
        """
        Initialize a Redis-backed vector store provider.

        Args:
            embedding_model (str): HuggingFace model for embeddings.
            url (str): Redis connection string.
            index (str): Name of the RediSearch index to use.
        """
        super().__init__(embedding_model)

        self.db = RedisVectorStore(
            index_name=index, embeddings=self.embeddings, redis_url=url
        )

        logger.info(
            "Connected to Redis at %s (index: %s)",
            url,
            index,
        )

    def add_documents(self, docs: List[Document]) -> None:
        """
        Add a list of documents to the Redis vector store.

        Args:
            docs (List[Document]): LangChain document chunks to embed and store.
        """
        self.db.add_documents(docs)
