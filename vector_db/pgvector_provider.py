from typing import List
from langchain_postgres import PGVector
from langchain_core.documents import Document
from vector_db.db_provider import DBProvider


class PGVectorProvider(DBProvider):
    """
    PGVector-based vector DB provider.

    Uses the `langchain_postgres.PGVector` integration to store
    document embeddings in a PostgreSQL-compatible backend with pgvector enabled.

    Args:
        url (str): PostgreSQL connection string (e.g. postgresql://user:pass@host:5432/db)
        collection_name (str): Name of the pgvector table or collection

    Example:
        >>> provider = PGVectorProvider(
        ...     url="postgresql://user:pass@localhost:5432/mydb",
        ...     collection_name="documents"
        ... )
        >>> provider.add_documents(chunks)
    """

    def __init__(self, url: str, collection_name: str):
        super().__init__()
        self.db = PGVector(
            connection=url,
            collection_name=collection_name,
            embeddings=self.embeddings,
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
