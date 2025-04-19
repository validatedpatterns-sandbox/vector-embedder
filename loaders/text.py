import logging
from pathlib import Path
from typing import List

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from unstructured.partition.auto import partition

from config import Config

logger = logging.getLogger(__name__)


class TextLoader:
    """
    Loads and semantically splits a list of general-purpose text documents
    (e.g., .txt, .md, .rst, .adoc) using Unstructured's local partitioning engine.

    This loader does not require an API key and works fully offline.

    Unstructured's `partition()` function breaks files into structured elements
    like titles, narrative text, lists, etc., which improves RAG chunk quality
    over naive fixed-size splitting.

    Attributes:
        config (Config): The job configuration, including chunk size and overlap.

    Example:
        >>> loader = TextLoader(config)
        >>> docs = loader.load([Path("README.md"), Path("guide.rst")])
        >>> print(docs[0].page_content)
    """

    def __init__(self, config: Config):
        self.config = config
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
        )

    def load(self, paths: List[Path]) -> List[Document]:
        """
        Loads and splits a list of text files into structured and chunked LangChain documents.

        Args:
            paths (List[Path]): List of file paths to load.

        Returns:
            List[Document]: Chunked LangChain Document objects with metadata.
        """
        all_chunks: List[Document] = []

        for path in paths:
            try:
                logger.info("Loading and partitioning: %s", path)
                elements = partition(filename=str(path), strategy="fast")

                docs = [
                    Document(
                        page_content=element.text,
                        metadata={
                            "source": str(path),
                            **(element.metadata.to_dict() if element.metadata else {}),
                        },
                    )
                    for element in elements
                    if hasattr(element, "text") and element.text
                ]

                chunks = self.splitter.split_documents(docs)
                all_chunks.extend(chunks)

            except Exception as e:
                logger.warning("Failed to load %s: %s", path, e)

        return all_chunks
