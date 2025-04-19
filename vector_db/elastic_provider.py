from typing import List

from langchain_core.documents import Document
from langchain_elasticsearch.vectorstores import ElasticsearchStore

from vector_db.db_provider import DBProvider


class ElasticProvider(DBProvider):
    """
    Elasticsearch-based vector DB provider using LangChain's ElasticsearchStore.

    Args:
        url (str): Full URL to the Elasticsearch cluster (e.g. http://localhost:9200)
        password (str): Authentication password for the cluster
        index (str): Index name to use for vector storage
        user (str): Username for Elasticsearch (default: "elastic")

    Example:
        >>> provider = ElasticProvider(
        ...     url="http://localhost:9200",
        ...     password="changeme",
        ...     index="docs",
        ...     user="elastic"
        ... )
        >>> provider.add_documents(chunks)
    """

    def __init__(self, url: str, password: str, index: str, user: str):
        super().__init__()
        self.db = ElasticsearchStore(
            embedding=self.embeddings,
            es_url=url,
            es_user=user,
            es_password=password,
            index_name=index,
        )

    def add_documents(self, docs: List[Document]) -> None:
        """
        Add documents to the Elasticsearch index.

        Args:
            docs (List[Document]): Chunked LangChain documents to index.
        """
        self.db.add_documents(docs)
