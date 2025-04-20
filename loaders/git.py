import logging
import shutil
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
    Loads and processes documents from Git repositories based on configured glob patterns.

    For each configured repository, this loader:
    - Clones or pulls the latest repo into a temporary folder
    - Applies the configured glob patterns to find matching files
    - Loads PDF files using PDFLoader
    - Loads supported text files using TextLoader
    - Returns all chunked LangChain Document objects (does NOT push to DB)

    Attributes:
        config (Config): Application config including temp paths, glob patterns, and chunking rules.

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
        Loads and chunks documents from all configured Git repos.

        This includes:
        - Cloning or updating each Git repo
        - Matching glob patterns to find relevant files
        - Loading and chunking documents using appropriate loaders

        Returns:
            List[Document]: Chunked LangChain documents from all matched files.
        """
        all_chunks: List[Document] = []

        for repo_entry in self.config.repo_sources:
            repo_url = repo_entry["repo"]
            globs = repo_entry.get("globs", [])
            repo_name = repo_url.rstrip("/").split("/")[-1].replace(".git", "")
            repo_path = self.base_path / repo_name

            self._ensure_repo_up_to_date(repo_url, repo_path)

            matched_files = self._collect_files(repo_path, globs)
            matched_files = [f for f in matched_files if f.is_file()]

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

    def _ensure_repo_up_to_date(self, url: str, dest: Path) -> None:
        if dest.exists():
            logger.info("Repo already cloned at %s, attempting pull...", dest)
            try:
                subprocess.run(
                    ["git", "-C", str(dest), "pull"],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                return
            except subprocess.CalledProcessError:
                logger.warning("git pull failed for %s, removing and recloning...", url)
                shutil.rmtree(dest)

        self._clone_repo(url, dest)

    def _clone_repo(self, url: str, dest: Path) -> None:
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
