import logging
from typing import List, Optional

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

from vector_db.db_provider import DBProvider

logger = logging.getLogger(__name__)


class QdrantProvider(DBProvider):
    """
    Vector database provider backed by Qdrant, using LangChain's QdrantVectorStore.

    This provider connects to a running Qdrant instance and stores embedded document
    vectors in a named collection. If the collection does not exist, it will be created
    automatically with COSINE distance.

    Attributes:
        client (QdrantClient): Low-level Qdrant client for managing collections.
        db (QdrantVectorStore): LangChain-compatible wrapper for vector operations.

    Args:
        embeddings (HuggingFaceEmbeddings): Pre-initialized HuggingFace embeddings instance.
        url (str): Base URL for the Qdrant service (e.g., "http://localhost:6333").
        collection (str): Name of the Qdrant collection to use.
        api_key (Optional[str]): Optional API key if authentication is required.

    Example:
        >>> from langchain_huggingface import HuggingFaceEmbeddings
        >>> from vector_db.qdrant_provider import QdrantProvider
        >>> embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
        >>> provider = QdrantProvider(
        ...     embeddings=embeddings,
        ...     url="http://localhost:6333",
        ...     collection="docs",
        ...     api_key=None
        ... )
        >>> provider.add_documents(chunks)
    """

    def __init__(
        self,
        embeddings: HuggingFaceEmbeddings,
        url: str,
        collection: str,
        api_key: Optional[str] = None,
    ):
        """
        Initialize the Qdrant vector DB provider.

        Args:
            embeddings (HuggingFaceEmbeddings): Embedding model instance.
            url (str): URL of the Qdrant instance.
            collection (str): Name of the collection to use or create.
            api_key (Optional[str]): Optional Qdrant API key.
        """
        super().__init__(embeddings)
        self.collection = collection
        self.url = url

        self.client = QdrantClient(
            url=url,
            api_key=api_key,
        )

        if not self._collection_exists():
            self._create_collection()

        self.db = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection,
            embedding=self.embeddings,
        )

        logger.info(
            "Connected to Qdrant at %s (collection: %s)", self.url, self.collection
        )

    def _collection_exists(self) -> bool:
        """
        Check if the Qdrant collection already exists.

        Returns:
            bool: True if the collection exists, False otherwise.
        """
        return self.client.collection_exists(self.collection)

    def _create_collection(self) -> None:
        """
        Create a new collection in Qdrant using the computed embedding length.
        """
        self.client.recreate_collection(
            collection_name=self.collection,
            vectors_config=VectorParams(
                size=self.embedding_length, distance=Distance.COSINE
            ),
        )

    def add_documents(self, docs: List[Document]) -> None:
        """
        Add a list of embedded documents to the Qdrant collection.

        Args:
            docs (List[Document]): LangChain documents to store in Qdrant.
        """
        self.db.add_documents(documents=docs)
