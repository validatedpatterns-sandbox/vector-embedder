from abc import ABC, abstractmethod
from typing import List

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings


class DBProvider(ABC):
    """
    Abstract base class for vector database providers.

    This class standardizes how vector databases are initialized and how documents
    are added to them. All concrete implementations (e.g., Qdrant, FAISS) must
    subclass `DBProvider` and implement the `add_documents()` method.

    Attributes:
        embeddings (Embeddings): An instance of HuggingFace embeddings based on the
                                 specified model.

    Args:
        embedding_model (str): HuggingFace-compatible model name to be used for computing
                               dense vector embeddings for documents.

    Example:
        >>> class MyProvider(DBProvider):
        ...     def add_documents(self, docs):
        ...         print(f"Would add {len(docs)} docs with model {self.embeddings.model_name}")

        >>> provider = MyProvider("BAAI/bge-small-en")
        >>> provider.add_documents([Document(page_content="Hello")])
    """

    def __init__(self, embedding_model: str) -> None:
        """
        Initialize a DB provider with a specific embedding model.

        Args:
            embedding_model (str): The HuggingFace model name to be used for generating embeddings.
        """
        self.embeddings: Embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

    @abstractmethod
    def add_documents(self, docs: List[Document]) -> None:
        """
        Add documents to the vector database.

        This method must be implemented by subclasses to define how documents
        (with or without precomputed embeddings) are stored in the backend vector DB.

        Args:
            docs (List[Document]): A list of LangChain `Document` objects to be embedded and added.
        """
        pass
