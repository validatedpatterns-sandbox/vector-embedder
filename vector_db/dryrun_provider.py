from typing import List
from langchain_core.documents import Document
from vector_db.db_provider import DBProvider


class DryRunProvider(DBProvider):
    """
    A mock DBProvider used in dry run mode.

    Instead of storing documents in a vector database, this provider prints the
    chunked documents to stdout. It is useful for debugging document loading,
    chunking, and metadata before committing to a real embedding operation.

    Example:
        >>> from vector_db.dry_run_provider import DryRunProvider
        >>> provider = DryRunProvider()
        >>> provider.add_documents(docs)  # docs is a List[Document]
    """

    def __init__(self):
        super().__init__()  # ensures embeddings are initialized

    def add_documents(self, docs: List[Document]) -> None:
        """
        Print chunked documents and metadata to stdout for debugging.

        Args:
            docs (List[Document]): The documents to preview in dry run mode.
        """
        print("\n=== Dry Run Output ===")
        for i, doc in enumerate(docs[:10]):
            print(f"\n--- Chunk {i + 1} ---")
            print(f"Content:\n{doc.page_content[:300]}")
            print(f"Metadata: {doc.metadata}")
        if len(docs) > 10:
            print(f"\n...and {len(docs) - 10} more chunk(s) not shown.")
        print("=== End Dry Run ===\n")
