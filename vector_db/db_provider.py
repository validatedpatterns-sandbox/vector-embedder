from abc import ABC, abstractmethod
from typing import List

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


class DBProvider(ABC):
    """
    Abstract base class for vector DB providers.
    Subclasses must implement `add_documents`.
    """

    def __init__(self) -> None:
        self.embeddings: Embeddings = HuggingFaceEmbeddings()

    @abstractmethod
    def add_documents(self, docs: List[Document]) -> None:
        """
        Add a list of documents (already embedded or to be embedded) to the vector store.
        Must be implemented by subclasses.
        """
        pass
