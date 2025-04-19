import logging
from typing import List

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document

from config import Config

logger = logging.getLogger(__name__)


class WebLoader:
    """
    Loads and semantically splits documents from a list of web URLs
    using LangChain's WebBaseLoader and a configurable chunking strategy.

    This loader does not embed documents directly — it returns them to be
    embedded by the caller.

    Attributes:
        config (Config): The application config containing chunk size and overlap.

    Example:
        >>> loader = WebLoader(config)
        >>> chunks = loader.load(["https://example.com/page1", "https://example.com/page2"])
    """

    def __init__(self, config: Config):
        self.config = config
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
        )

    def load(self, urls: List[str]) -> List[Document]:
        """
        Load and chunk documents from the given list of URLs.

        Args:
            urls (List[str]): Web URLs to load.

        Returns:
            List[Document]: A list of chunked LangChain documents.
        """
        if not urls:
            logger.warning("No URLs provided to WebLoader.")
            return []

        logger.info("Loading web documents from %d URL(s)...", len(urls))
        for url in urls:
            logger.debug(" - %s", url)

        try:
            loader = WebBaseLoader(urls)
            docs = loader.load()
        except Exception:
            logger.exception("Failed to load URLs via WebBaseLoader")
            raise

        logger.info(
            "Splitting %d document(s) with chunk size %s and overlap %s",
            len(docs),
            self.config.chunk_size,
            self.config.chunk_overlap,
        )

        return self.splitter.split_documents(docs)
