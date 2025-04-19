import logging
import subprocess
from pathlib import Path
from typing import List

from langchain_core.documents import Document

from config import Config
from loaders.pdf import PDFLoader
from loaders.text import TextLoader

logger = logging.getLogger(__name__)


class GitLoader:
    """
    Clones repositories and loads documents based on glob patterns using PDF and text loaders.

    For each repository defined in `config.repo_sources`, this loader:
        - Clones the repo to a local folder in TEMP_DIR/source_repo/
        - Resolves the configured globs relative to the repo root
        - Loads and chunks `.pdf` files using PDFLoader
        - Loads and chunks all other supported text files using TextLoader

    This loader returns all chunked `Document` objects so the caller can decide
    how and when to add them to a vector store.

    Attributes:
        config (Config): The configuration object containing repo info, chunk settings, etc.

    Example:
        >>> config = Config.load()
        >>> loader = GitLoader(config)
        >>> chunks = loader.load()
        >>> config.db_provider.add_documents(chunks)
    """

    def __init__(self, config: Config):
        self.config = config
        self.base_path = Path(config.temp_dir) / "repo_sources"
        self.pdf_loader = PDFLoader(config)
        self.text_loader = TextLoader(config)

    def load(self) -> List[Document]:
        """
        Clones all configured repos, applies glob patterns, loads and chunks matched documents.

        Returns:
            List[Document]: All chunked documents loaded from the matched files in all repositories.

        Raises:
            RuntimeError: If cloning or file loading fails.
        """
        all_chunks: List[Document] = []

        for repo_entry in self.config.repo_sources:
            repo_url = repo_entry["repo"]
            globs = repo_entry.get("globs", [])
            repo_name = repo_url.rstrip("/").split("/")[-1].replace(".git", "")
            repo_path = self.base_path / repo_name

            self._clone_repo(repo_url, repo_path)

            matched_files = self._collect_files(repo_path, globs)

            pdf_files = [f for f in matched_files if f.suffix.lower() == ".pdf"]
            text_files = [f for f in matched_files if f.suffix.lower() != ".pdf"]

            if pdf_files:
                logger.info("Loading %d PDF file(s) from %s", len(pdf_files), repo_url)
                all_chunks.extend(self.pdf_loader.load(pdf_files))

            if text_files:
                logger.info(
                    "Loading %d text file(s) from %s", len(text_files), repo_url
                )
                all_chunks.extend(self.text_loader.load(text_files))

        return all_chunks

    def _clone_repo(self, url: str, dest: Path) -> None:
        if dest.exists():
            logger.info("Repo already cloned at %s, skipping", dest)
            return

        logger.info("Cloning repository %s to %s", url, dest)
        try:
            subprocess.run(
                ["git", "clone", "--depth", "1", url, str(dest)],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except subprocess.CalledProcessError as e:
            logger.error("Failed to clone %s: %s", url, e)
            raise RuntimeError(f"Failed to clone repo: {url}") from e

    def _collect_files(self, base: Path, patterns: List[str]) -> List[Path]:
        matched: List[Path] = []

        for pattern in patterns:
            found = list(base.glob(pattern))
            if not found:
                logger.warning("No files matched pattern '%s' in %s", pattern, base)
            matched.extend(found)

        logger.info("Matched %d files in %s", len(matched), base)
        return matched
