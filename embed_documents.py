#!/usr/bin/env python

"""
embed_documents.py

Main entry point for embedding documents into a vector database.

This script performs the following operations:
1. Loads configuration and initializes DB provider and loaders.
2. Fetches and embeds Git-sourced documents (Markdown, PDFs, etc.).
3. Fetches and embeds web documents (HTML and PDFs).
4. Chunks all documents and indexes them into the configured vector store.

This tool is designed for use in pipelines or manual indexing workflows.

Usage:
    $ python embed_documents.py

Environment:
    Requires a valid .env file or environment variables defined for:
    - DB_TYPE, EMBEDDING_MODEL, TEMP_DIR
    - CHUNK_SIZE, CHUNK_OVERLAP, LOG_LEVEL
    - WEB_SOURCES, REPO_SOURCES
    - Plus additional DB-specific variables based on DB_TYPE

Example:
    $ DB_TYPE=QDRANT EMBEDDING_MODEL=BAAI/bge-large-en-v1.5 \
      CHUNK_SIZE=20480 CHUNK_OVERLAP=2048 \
      TEMP_DIR=/tmp EMBEDDING_MODEL=... \
      python embed_documents.py
"""

import logging
import sys
from pathlib import Path

import requests

from config import Config
from loaders.git import GitLoader
from loaders.pdf import PDFLoader
from loaders.web import WebLoader

config = Config.load()
logger = logging.getLogger(__name__)


def _fail_and_exit(message: str, exc: Exception) -> None:
    """
    Log an error with full traceback and raise the exception.

    Args:
        message (str): Contextual message to log with the error.
        exc (Exception): The exception to raise.

    This utility is used to ensure proper logging and failure behavior
    across all critical stages of the embedding job.
    """
    logger.error("%s: %s", message, exc, exc_info=True)
    raise exc


def _process_git_documents() -> None:
    """Process Git-based document sources and add to vector DB."""
    if not config.repo_sources:
        return

    logger.info("Starting Git-based document embedding...")
    try:
        git_loader = GitLoader(config)
        git_chunks = git_loader.load()

        if git_chunks:
            logger.info("Adding %d Git document chunks to vector DB", len(git_chunks))
            config.db_provider.add_documents(git_chunks)
        else:
            logger.info("No documents found in Git sources.")
    except Exception as e:
        _fail_and_exit("Failed during Git document processing", e)


def _process_html_documents(html_urls: list) -> None:
    """Process HTML web documents and add to vector DB."""
    if not html_urls:
        return

    logger.info("Starting HTML-based web document embedding...")
    try:
        web_loader = WebLoader(config)
        web_chunks = web_loader.load(html_urls)

        if web_chunks:
            logger.info("Adding %d HTML web chunks to vector DB", len(web_chunks))
            config.db_provider.add_documents(web_chunks)
        else:
            logger.info("No chunks produced from HTML URLs.")
    except Exception as e:
        _fail_and_exit("Failed during HTML web document processing", e)


def _process_pdf_documents(pdf_urls: list) -> None:
    """Download and process PDF documents from web URLs and add to vector DB."""
    if not pdf_urls:
        return

    logger.info("Downloading PDF documents from web URLs...")
    pdf_dir = Path(config.temp_dir) / "web_pdfs"
    pdf_dir.mkdir(parents=True, exist_ok=True)

    downloaded_files = []
    for url in pdf_urls:
        try:
            response = requests.get(url)
            response.raise_for_status()

            filename = Path(url.split("/")[-1])
            file_path = pdf_dir / filename
            with open(file_path, "wb") as f:
                f.write(response.content)

            logger.info("Downloaded: %s", file_path)
            downloaded_files.append(file_path)
        except Exception as e:
            _fail_and_exit(f"Failed to download {url}", e)

    if downloaded_files:
        try:
            pdf_loader = PDFLoader(config)
            pdf_chunks = pdf_loader.load(downloaded_files)

            if pdf_chunks:
                logger.info("Adding %d PDF web chunks to vector DB", len(pdf_chunks))
                config.db_provider.add_documents(pdf_chunks)
            else:
                logger.info("No chunks produced from downloaded PDFs.")
        except Exception as e:
            _fail_and_exit("Failed during PDF web document processing", e)


def main() -> None:
    """
    Main embedding workflow for Git, HTML, and PDF sources.

    Steps:
        1. Load and chunk files from configured Git repos, if any.
        2. Load and chunk HTML documents from web sources.
        3. Download, load, and chunk remote PDF files.
        4. Store all chunks into the configured vector DB provider.

    All errors are logged with traceback and will stop execution via `_fail_and_exit`.
    """
    _process_git_documents()

    pdf_urls = [url for url in config.web_sources if url.lower().endswith(".pdf")]
    html_urls = [url for url in config.web_sources if not url.lower().endswith(".pdf")]

    _process_html_documents(html_urls)
    _process_pdf_documents(pdf_urls)

    logger.info("Embedding job complete.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.critical("Fatal error: %s", e, exc_info=True)
        sys.exit(1)
