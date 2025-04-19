#!/usr/bin/env python

import logging

from config import Config
from loaders.git import GitLoader
from loaders.web import WebLoader

# Initialize logging
config = Config.load()
logging.basicConfig(level=config.log_level)
logger = logging.getLogger(__name__)

# Run Git document embedding if sources provided
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

# Run Web document embedding if URLs provided
if config.web_sources:
    logger.info("Starting Web-based document embedding...")
    try:
        web_loader = WebLoader(config)
        web_chunks = web_loader.load(config.web_sources)

        if web_chunks:
            logger.info(
                "Adding %d document chunks from Web to vector DB", len(web_chunks)
            )
            config.db_provider.add_documents(web_chunks)
        else:
            logger.info("No documents returned from provided URLs.")
    except Exception:
        logger.exception("Failed during Web document processing")

logger.info("Embedding job complete.")
