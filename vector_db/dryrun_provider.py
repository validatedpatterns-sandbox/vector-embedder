from typing import List

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from vector_db.db_provider import DBProvider


class DryRunProvider(DBProvider):
    """
    A mock vector DB provider for debugging document loading and chunking.

    `DryRunProvider` does not persist any documents or perform actual embedding.
    It prints a preview of the documents and their metadata to stdout, allowing users
    to validate chunking, structure, and metadata before pushing to a production vector store.

    Attributes:
        embeddings (HuggingFaceEmbeddings): HuggingFace embedding instance, used for interface consistency.
        embedding_length (int): Dimensionality of embeddings (computed for validation, not used).

    Args:
        embeddings (HuggingFaceEmbeddings): A HuggingFace embedding model instance.

    Example:
        >>> from langchain_core.documents import Document
        >>> from langchain_huggingface import HuggingFaceEmbeddings
        >>> from vector_db.dryrun_provider import DryRunProvider
        >>> embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en")
        >>> provider = DryRunProvider(embeddings)
        >>> docs = [Document(page_content="Hello world", metadata={"source": "test.txt"})]
        >>> provider.add_documents(docs)
    """

    def __init__(self, embeddings: HuggingFaceEmbeddings):
        """
        Initialize the dry run provider with a placeholder embedding model.

        Args:
            embeddings (HuggingFaceEmbeddings): A HuggingFace embedding model (used for compatibility).
        """
        super().__init__(embeddings)

    def add_documents(self, docs: List[Document]) -> None:
        """
        Print chunked documents and metadata to stdout for inspection.

        This method displays the first 10 document chunks, including the start
        of their page content and associated metadata.

        Args:
            docs (List[Document]): A list of LangChain documents to preview.
        """
        print("\n=== Dry Run Output ===")
        for i, doc in enumerate(docs[:10]):
            print(f"\n--- Chunk {i + 1} ---")
            print(f"Content:\n{doc.page_content[:300]}")
            print(f"Metadata: {doc.metadata}")
        if len(docs) > 10:
            print(f"\n...and {len(docs) - 10} more chunk(s) not shown.")
        print("=== End Dry Run ===\n")
