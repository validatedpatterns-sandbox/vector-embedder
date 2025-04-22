import logging
from pathlib import Path
from typing import List

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

from config import Config

logger = logging.getLogger(__name__)


class PDFLoader:
    """
    Loads and splits a list of PDF files into chunked LangChain `Document` objects.

    This loader:
    - Uses `PyPDFLoader` to convert each page of a PDF into text
    - Applies `RecursiveCharacterTextSplitter` to split pages into smaller overlapping chunks
    - Annotates each chunk with metadata including `source` and `chunk_id`

    Attributes:
        config (Config): Configuration object that specifies chunk size and overlap.

    Example:
        >>> loader = PDFLoader(config)
        >>> docs = loader.load([Path("whitepaper.pdf"), Path("spec.pdf")])
        >>> print(docs[0].metadata)
        {'source': 'whitepaper.pdf', 'chunk_id': 0}
    """

    def __init__(self, config: Config):
        """
        Initialize the PDFLoader with a given configuration.

        Args:
            config (Config): Contains chunking parameters (chunk_size, chunk_overlap).
        """
        self.config = config
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
        )

    def load(self, paths: List[Path]) -> List[Document]:
        """
        Loads and chunks the content of each PDF file.

        For each PDF:
            - Extracts per-page content using `PyPDFLoader`
            - Splits text into chunks optimized for retrieval
            - Annotates each chunk with its source path and chunk index

        Args:
            paths (List[Path]): List of PDF file paths to process.

        Returns:
            List[Document]: List of chunked documents with metadata attached.
        """
        all_chunks: List[Document] = []

        for path in paths:
            try:
                logger.info("Loading PDF: %s", path)
                pages = PyPDFLoader(str(path)).load()
                chunks = self.splitter.split_documents(pages)

                for cid, ch in enumerate(chunks):
                    ch.metadata.setdefault("source", str(path))
                    ch.metadata["chunk_id"] = cid

                all_chunks.extend(chunks)

            except Exception as e:
                logger.warning("Failed to load PDF %s: %s", path, e)

        return all_chunks
