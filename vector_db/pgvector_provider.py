"""PostgreSQL with pgvector extension vector database provider implementation."""

import logging
from typing import List
from urllib.parse import urlparse

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres import PGEngine, PGVectorStore

from vector_db.db_provider import DBProvider

logger = logging.getLogger(__name__)


class PGVectorProvider(DBProvider):
    """
    Vector database provider backed by PostgreSQL with pgvector extension.

    This provider uses LangChain's `PGVector` integration to store and query
    embedded documents in a PostgreSQL-compatible database. It requires a working
    `pgvector` extension in the target database.

    Attributes:
        db (PGVectorStore): LangChain-compatible PGVector client for vector storage.
        embeddings (HuggingFaceEmbeddings): HuggingFace model for generating document vectors.

    Args:
        embeddings (HuggingFaceEmbeddings): HuggingFace embeddings instance.
        url (str): PostgreSQL connection string (e.g., "postgresql://user:pass@host:5432/db").
        collection_name (str): Name of the table/collection used for storing vectors.

    Example:
        >>> from langchain_huggingface import HuggingFaceEmbeddings
        >>> from vector_db.pgvector_provider import PGVectorProvider
        >>> embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
        >>> provider = PGVectorProvider(
        ...     embeddings=embeddings,
        ...     url="postgresql://user:pass@localhost:5432/vector_db",
        ...     collection_name="rag_chunks"
        ... )
        >>> provider.add_documents(docs)
    """

    def __init__(
        self,
        embeddings: HuggingFaceEmbeddings,
        url: str,
        collection_name: str,
    ):
        """
        Initialize a PGVectorProvider for use with PostgreSQL.

        Args:
            embeddings (HuggingFaceEmbeddings): Embedding model for vector generation.
            url (str): PostgreSQL connection string with pgvector enabled.
            collection_name (str): Name of the vector table in the database.
        """
        super().__init__(embeddings)

        engine = PGEngine.from_connection_string(url)
        engine.init_vectorstore_table(collection_name, self.embedding_length)

        self.db = PGVectorStore.create_sync(engine, self.embeddings, collection_name)

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
        Store a list of documents in the PGVector collection.

        This embeds documents using the provided model and persists them
        to the PostgreSQL backend. Null bytes (\\x00) are stripped to prevent DB errors.

        Args:
            docs (List[Document]): Chunked LangChain documents to store.
        """
        for doc in docs:
            doc.page_content = doc.page_content.replace("\x00", "")
        self.db.add_documents(docs)
