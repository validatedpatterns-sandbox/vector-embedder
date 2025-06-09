from abc import ABC, abstractmethod
from typing import List

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings


class DBProvider(ABC):
    """
    Abstract base class for vector database providers.

    This class standardizes how vector databases are initialized and how documents
    are added to them. All concrete implementations (e.g., Qdrant, Redis) must
    subclass `DBProvider` and implement the `add_documents()` method.

    Attributes:
        embeddings (HuggingFaceEmbeddings): An instance of HuggingFace embeddings.
        embedding_length (int): Dimensionality of the embedding vector.

    Args:
        embeddings (HuggingFaceEmbeddings): A preconfigured HuggingFaceEmbeddings instance.

    Example:
        >>> class MyProvider(DBProvider):
        ...     def add_documents(self, docs):
        ...         print(f"Would add {len(docs)} docs with vector size {self.embedding_length}")

        >>> embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en")
        >>> provider = MyProvider(embeddings)
        >>> provider.add_documents([Document(page_content="Hello")])
    """

    def __init__(self, embeddings: HuggingFaceEmbeddings) -> None:
        """
        Initialize a DB provider with a HuggingFaceEmbeddings instance.

        Args:
            embeddings (HuggingFaceEmbeddings): The embeddings object used for vectorization.
        """
        self.embeddings: HuggingFaceEmbeddings = embeddings
        self.embedding_length: int = len(self.embeddings.embed_query("query"))

    @abstractmethod
    def add_documents(self, docs: List[Document]) -> None:
        """
        Add documents to the vector database.

        This method must be implemented by subclasses to define how documents
        are embedded and stored in the backend vector DB.

        Args:
            docs (List[Document]): A list of LangChain `Document` objects to be embedded and added.
        """
        pass
