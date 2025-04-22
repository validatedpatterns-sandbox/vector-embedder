import logging
from typing import Dict, List

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document

from config import Config

logger = logging.getLogger(__name__)


class WebLoader:
    """
    Loads and chunks content from a list of web URLs using LangChain's `WebBaseLoader`.

    This loader:
    - Fetches HTML or text content from each URL
    - Applies recursive text chunking based on the configured chunk size and overlap
    - Annotates each chunk with metadata including `source` (URL) and `chunk_id`
      for downstream neighbor expansion in retrieval tasks

    Attributes:
        config (Config): Configuration object containing chunk size and overlap.

    Example:
        >>> loader = WebLoader(config)
        >>> chunks = loader.load(["https://example.com/intro", "https://example.com/docs"])
        >>> print(chunks[0].metadata)
        {'source': 'https://example.com/intro', 'chunk_id': 0}
    """

    def __init__(self, config: Config):
        """
        Initialize the WebLoader with a given configuration.

        Args:
            config (Config): Configuration object with chunking parameters.
        """
        self.config = config
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
        )

    def load(self, urls: List[str]) -> List[Document]:
        """
        Loads and splits documents from the given list of URLs.

        Steps:
            - Downloads the web pages
            - Splits content into semantic or character-based chunks
            - Adds `source` and `chunk_id` to each chunk for traceability

        Args:
            urls (List[str]): List of web URLs to fetch and process.

        Returns:
            List[Document]: A list of chunked `Document` objects with metadata.
        """
        if not urls:
            logger.warning("WebLoader called with empty URL list.")
            return []

        logger.info("Loading %d web document(s)…", len(urls))
        try:
            docs = WebBaseLoader(urls).load()
        except Exception:
            logger.exception("Failed to fetch one or more URLs")
            raise

        chunks = self.splitter.split_documents(docs)

        # Assign unique chunk_id per source URL
        per_source_counter: Dict[str, int] = {}
        for ch in chunks:
            src = ch.metadata.get("source") or ch.metadata.get("url") or "unknown"
            ch.metadata["source"] = src
            ch.metadata["chunk_id"] = per_source_counter.setdefault(src, 0)
            per_source_counter[src] += 1

        logger.info(
            "Produced %d web chunks (avg %.0f chars)",
            len(chunks),
            sum(len(c.page_content) for c in chunks) / max(1, len(chunks)),
        )

        return chunks
