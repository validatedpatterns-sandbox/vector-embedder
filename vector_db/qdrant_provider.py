import logging
from typing import List, Optional

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

from vector_db.db_provider import DBProvider

logger = logging.getLogger(__name__)


class QdrantProvider(DBProvider):
    """
    Qdrant-based vector DB provider using LangChain's QdrantVectorStore.

    Args:
        url (str): Base URL of the Qdrant service (e.g., http://localhost:6333)
        collection (str): Name of the vector collection to use or create
        api_key (Optional[str]): API key if authentication is required (optional)

    This provider will create the collection if it does not already exist.

    Example:
        >>> provider = QdrantProvider(
        ...     url="http://localhost:6333",
        ...     collection="embedded_docs",
        ...     api_key=None
        ... )
        >>> provider.add_documents(docs)
    """

    def __init__(self, url: str, collection: str, api_key: Optional[str] = None):
        super().__init__()
        self.collection = collection
        self.url = url

        self.client = QdrantClient(
            url=url,
            api_key=api_key or None,
        )

        if not self._collection_exists():
            self._create_collection()

        self.db = QdrantVectorStore(
            client=self.client,
            collection_name=collection,
            embedding=self.embeddings,
        )

        logger.info(
            "Connected to Qdrant instance at %s (collection: %s)",
            self.url,
            self.collection,
        )

    def _collection_exists(self) -> bool:
        return self.client.collection_exists(self.collection)

    def _create_collection(self) -> None:
        vector_size = len(self.embeddings.embed_query("test"))
        self.client.recreate_collection(
            collection_name=self.collection,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )

    def add_documents(self, docs: List[Document]) -> None:
        """
        Add documents to the Qdrant vector store.

        Args:
            docs (List[Document]): Chunked LangChain documents to index.
        """
        self.db.add_documents(documents=docs)
