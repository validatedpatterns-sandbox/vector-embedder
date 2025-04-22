import logging
from typing import List

from langchain_core.documents import Document
from langchain_elasticsearch.vectorstores import ElasticsearchStore

from vector_db.db_provider import DBProvider

logger = logging.getLogger(__name__)


class ElasticProvider(DBProvider):
    """
    Vector database provider backed by Elasticsearch using LangChain's ElasticsearchStore.

    This provider allows storing and querying vectorized documents in an Elasticsearch
    cluster. Documents are embedded using a HuggingFace model and stored with associated
    metadata in the specified index.

    Attributes:
        db (ElasticsearchStore): LangChain-compatible wrapper around Elasticsearch vector storage.
        embeddings (Embeddings): HuggingFace embedding model for generating document vectors.

    Args:
        embedding_model (str): HuggingFace model name for computing embeddings.
        url (str): Full URL to the Elasticsearch cluster (e.g. "http://localhost:9200").
        password (str): Password for the Elasticsearch user.
        index (str): The index name where documents will be stored.
        user (str): Elasticsearch username (default is typically "elastic").

    Example:
        >>> from vector_db.elastic_provider import ElasticProvider
        >>> provider = ElasticProvider(
        ...     embedding_model="BAAI/bge-small-en",
        ...     url="http://localhost:9200",
        ...     password="changeme",
        ...     index="rag-docs",
        ...     user="elastic"
        ... )
        >>> provider.add_documents(docs)
    """

    def __init__(
        self,
        embedding_model: str,
        url: str,
        password: str,
        index: str,
        user: str,
    ):
        """
        Initialize an Elasticsearch-based vector DB provider.

        Args:
            embedding_model (str): The model name for computing embeddings.
            url (str): Full URL of the Elasticsearch service.
            password (str): Elasticsearch user's password.
            index (str): Name of the Elasticsearch index to use.
            user (str): Elasticsearch username (e.g., "elastic").
        """
        super().__init__(embedding_model)

        self.db = ElasticsearchStore(
            embedding=self.embeddings,
            es_url=url,
            es_user=user,
            es_password=password,
            index_name=index,
        )

        logger.info("Connected to Elasticsearch at %s (index: %s)", url, index)

    def add_documents(self, docs: List[Document]) -> None:
        """
        Add a batch of LangChain documents to the Elasticsearch index.

        Each document will be embedded using the configured model and stored
        in the specified index with any associated metadata.

        Args:
            docs (List[Document]): List of documents to index.
        """
        self.db.add_documents(docs)
