"""Elasticsearch vector database provider implementation."""

import logging
from typing import List

from langchain_core.documents import Document
from langchain_elasticsearch.vectorstores import ElasticsearchStore
from langchain_huggingface import HuggingFaceEmbeddings

from vector_db.db_provider import DBProvider

logger = logging.getLogger(__name__)


class ElasticProvider(DBProvider):
    """
    Vector database provider backed by Elasticsearch using LangChain's ElasticsearchStore.

    This provider stores and queries vectorized documents in an Elasticsearch cluster.
    Documents are embedded using the provided HuggingFace embeddings model and stored
    with associated metadata in the specified index.

    Attributes:
        db (ElasticsearchStore): LangChain-compatible Elasticsearch vector store.
        embeddings (HuggingFaceEmbeddings): HuggingFace embedding model instance.

    Args:
        embeddings (HuggingFaceEmbeddings): Pre-initialized embeddings instance.
        url (str): Full URL to the Elasticsearch cluster (e.g., "http://localhost:9200").
        password (str): Password for the Elasticsearch user.
        index (str): The index name where documents will be stored.
        user (str): Elasticsearch username (default is typically "elastic").

    Example:
        >>> from langchain_huggingface import HuggingFaceEmbeddings
        >>> from vector_db.elastic_provider import ElasticProvider
        >>> embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en")
        >>> provider = ElasticProvider(
        ...     embeddings=embeddings,
        ...     url="http://localhost:9200",
        ...     password="changeme",
        ...     index="rag-docs",
        ...     user="elastic"
        ... )
        >>> provider.add_documents(docs)
    """

    def __init__(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        embeddings: HuggingFaceEmbeddings,
        url: str,
        password: str,
        index: str,
        user: str,
    ):
        """
        Initialize an Elasticsearch-based vector DB provider.

        Args:
            embeddings (HuggingFaceEmbeddings): HuggingFace embeddings instance.
            url (str): Full URL of the Elasticsearch service.
            password (str): Elasticsearch user's password.
            index (str): Name of the Elasticsearch index to use.
            user (str): Elasticsearch username (e.g., "elastic").
        """
        super().__init__(embeddings)

        # We use an incresed timeout since resources are constrained in CI environments
        es_params = {
            "timeout": 600,
        }

        self.db = ElasticsearchStore(
            embedding=self.embeddings,
            es_url=url,
            es_user=user,
            es_password=password,
            index_name=index,
            es_params=es_params,
        )

        logger.info("Connected to Elasticsearch at %s (index: %s)", url, index)

    def add_documents(self, docs: List[Document]) -> None:
        """
        Add a batch of LangChain documents to the Elasticsearch index.

        Each document is embedded using the provided model and stored
        in the specified index with its associated metadata.

        Args:
            docs (List[Document]): List of documents to index.
        """
        batch_size = 50
        for i in range(0, len(docs), batch_size):
            batch = docs[i : i + batch_size]
            try:
                self.db.add_documents(batch)
            except Exception:
                logger.exception("Failed to insert batch starting at index %s", i)
                raise
