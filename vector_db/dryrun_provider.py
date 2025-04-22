from typing import List

from langchain_core.documents import Document

from vector_db.db_provider import DBProvider


class DryRunProvider(DBProvider):
    """
    A mock vector DB provider for debugging document loading and chunking.

    `DryRunProvider` does not persist any documents or perform embedding operations.
    Instead, it prints a preview of the documents and their metadata to stdout,
    allowing users to validate chunking, structure, and metadata before pushing
    to a production vector store.

    Useful for development, testing, or understanding how your documents are
    being processed.

    Attributes:
        embeddings (Embeddings): HuggingFace embedding model for compatibility.

    Args:
        embedding_model (str): The model name to initialize HuggingFaceEmbeddings.
                               Used only for compatibility — no embeddings are generated.

    Example:
        >>> from langchain_core.documents import Document
        >>> provider = DryRunProvider("BAAI/bge-small-en")
        >>> docs = [Document(page_content="Hello world", metadata={"source": "test.txt"})]
        >>> provider.add_documents(docs)
    """

    def __init__(self, embedding_model: str):
        """
        Initialize the dry run provider with a placeholder embedding model.

        Args:
            embedding_model (str): The name of the embedding model (used for interface consistency).
        """
        super().__init__(embedding_model)

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
