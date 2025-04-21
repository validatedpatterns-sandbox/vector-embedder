from abc import ABC, abstractmethod
from typing import List

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings


class DBProvider(ABC):
    """
    Abstract base class for vector DB providers.
    Subclasses must implement `add_documents`.

    Args:
        embedding_model (str): Embedding model to use
    """

    def __init__(self, embedding_model: str) -> None:
        self.embeddings: Embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

    @abstractmethod
    def add_documents(self, docs: List[Document]) -> None:
        """
        Add a list of documents (already embedded or to be embedded) to the vector store.
        Must be implemented by subclasses.
        """
        pass
