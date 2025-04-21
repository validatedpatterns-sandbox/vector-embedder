import logging
from typing import List
from urllib.parse import urlparse

from langchain_core.documents import Document
from langchain_postgres import PGVector

from vector_db.db_provider import DBProvider

logger = logging.getLogger(__name__)


class PGVectorProvider(DBProvider):
    """
    PGVector-based vector DB provider.

    Uses the `langchain_postgres.PGVector` integration to store
    document embeddings in a PostgreSQL-compatible backend with pgvector enabled.

    Args:
        embedding_model (str): Embedding model to use
        url (str): PostgreSQL connection string (e.g. postgresql://user:pass@host:5432/db)
        collection_name (str): Name of the pgvector table or collection

    Example:
        >>> provider = PGVectorProvider(
        ...     embedding_model="sentence-transformers/all-mpnet-base-v2",
        ...     url="postgresql://user:pass@localhost:5432/mydb",
        ...     collection_name="documents"
        ... )
        >>> provider.add_documents(chunks)
    """

    def __init__(self, embedding_model: str, url: str, collection_name: str):
        super().__init__(embedding_model)

        self.db = PGVector(
            connection=url,
            collection_name=collection_name,
            embeddings=self.embeddings,
        )

        parsed = urlparse(url)
        postgres_location = (
            f"{parsed.hostname}:{parsed.port or 5432}/{parsed.path.lstrip('/')}"
        )
        logger.info(
            "Connected to PGVector at %s (collection: %s)",
            postgres_location,
            collection_name,
        )

    def add_documents(self, docs: List[Document]) -> None:
        """
        Add a list of documents to the pgvector-backed vector store.

        Args:
            docs (List[Document]): LangChain document chunks to embed and store.

        Notes:
            - Null bytes (`\\x00`) are removed from content to avoid PostgreSQL errors.
        """
        for doc in docs:
            doc.page_content = doc.page_content.replace("\x00", "")
        self.db.add_documents(docs)
