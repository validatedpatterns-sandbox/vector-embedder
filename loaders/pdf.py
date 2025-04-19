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
    Loads and splits a list of PDF documents using PyPDFLoader.

    Each PDF is processed individually, then split into smaller chunks
    using RecursiveCharacterTextSplitter to optimize for RAG and vector DB ingestion.

    Attributes:
        config (Config): Global configuration for chunking and db connection.

    Example:
        >>> loader = PDFLoader(config)
        >>> chunks = loader.load([Path("paper.pdf"), Path("spec.pdf")])
        >>> print(chunks[0].page_content)
    """

    def __init__(self, config: Config):
        self.config = config
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
        )

    def load(self, paths: List[Path]) -> List[Document]:
        """
        Loads and splits a list of PDF files.

        Args:
            paths (List[Path]): List of PDF file paths.

        Returns:
            List[Document]: A list of chunked LangChain Document objects.
        """
        all_chunks: List[Document] = []

        for path in paths:
            try:
                logger.info("Loading PDF: %s", path)
                loader = PyPDFLoader(str(path))
                docs = loader.load()
                chunks = self.splitter.split_documents(docs)
                all_chunks.extend(chunks)
            except Exception as e:
                logger.warning("Failed to load PDF file %s: %s", path, e)

        return all_chunks
