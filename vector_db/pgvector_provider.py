import logging
from typing import List
from urllib.parse import urlparse

from langchain_core.documents import Document
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
        db (PGVector): LangChain-compatible PGVector client for vector storage.
        embeddings (Embeddings): HuggingFace model for generating document vectors.

    Args:
        embedding_model (str): The model name to use for computing embeddings.
        url (str): PostgreSQL connection string (e.g. "postgresql://user:pass@host:5432/db").
        collection_name (str): Name of the table/collection used for storing vectors.
        embedding_length (int): Dimensionality of the embeddings (e.g., 768 for all-mpnet-base-v2).

    Example:
        >>> from vector_db.pgvector_provider import PGVectorProvider
        >>> provider = PGVectorProvider(
        ...     embedding_model="BAAI/bge-base-en-v1.5",
        ...     url="postgresql://user:pass@localhost:5432/vector_db",
        ...     collection_name="rag_chunks",
        ...     embedding_length=768
        ... )
        >>> provider.add_documents(docs)
    """

    def __init__(
        self,
        embedding_model: str,
        url: str,
        collection_name: str,
        embedding_length: int,
    ):
        """
        Initialize a PGVectorProvider for use with PostgreSQL.

        Args:
            embedding_model (str): HuggingFace model used for embedding chunks.
            url (str): Connection string to PostgreSQL with pgvector enabled.
            collection_name (str): Name of the vector table in the database.
        """
        super().__init__(embedding_model)

        engine = PGEngine.from_connection_string(url)
        engine.init_vectorstore_table(collection_name, embedding_length)

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

        This will embed the documents using the configured model and persist them
        to the PostgreSQL backend. Any null bytes (\\x00) are removed from text to
        prevent PostgreSQL errors.

        Args:
            docs (List[Document]): Chunked LangChain documents to store.
        """
        for doc in docs:
            doc.page_content = doc.page_content.replace("\x00", "")
        self.db.add_documents(docs)
