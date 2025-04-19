#!/usr/bin/env python

import logging
from pathlib import Path

import requests

from config import Config
from loaders.git import GitLoader
from loaders.pdf import PDFLoader
from loaders.web import WebLoader

# Load environment config
config = Config.load()

# Configure logging
logging.basicConfig(level=config.log_level)
logger = logging.getLogger(__name__)

#  Git-based embedding
if config.repo_sources:
    logger.info("Starting Git-based document embedding...")
    try:
        git_loader = GitLoader(config)
        git_chunks = git_loader.load()

        if git_chunks:
            logger.info(
                "Adding %d document chunks from Git to vector DB", len(git_chunks)
            )
            config.db_provider.add_documents(git_chunks)
        else:
            logger.info("No documents found in Git sources.")
    except Exception:
        logger.exception("Failed during Git document processing")

#  Separate Web URLs by type
pdf_urls = [url for url in config.web_sources if url.lower().endswith(".pdf")]
html_urls = [url for url in config.web_sources if not url.lower().endswith(".pdf")]

#  HTML URL embedding
if html_urls:
    logger.info("Starting HTML-based web document embedding...")
    try:
        web_loader = WebLoader(config)
        web_chunks = web_loader.load(html_urls)

        if web_chunks:
            logger.info("Adding %d HTML chunks to vector DB", len(web_chunks))
            config.db_provider.add_documents(web_chunks)
        else:
            logger.info("No chunks produced from HTML URLs.")
    except Exception:
        logger.exception("Failed during HTML web document processing")

#  PDF URL embedding
if pdf_urls:
    logger.info("Processing PDF documents from web URLs...")

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
            logger.exception("Failed to download %s: %s", url, e)

    if downloaded_files:
        try:
            pdf_loader = PDFLoader(config)
            pdf_chunks = pdf_loader.load(downloaded_files)

            if pdf_chunks:
                logger.info("Adding %d PDF chunks to vector DB", len(pdf_chunks))
                config.db_provider.add_documents(pdf_chunks)
            else:
                logger.info("No chunks produced from downloaded PDFs.")
        except Exception:
            logger.exception("Failed during PDF web document processing")

logger.info("Embedding job complete.")
