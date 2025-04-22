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
    Loads and chunks documents from Git repositories based on configured glob patterns.

    For each Git repository specified in `config.repo_sources`, this loader:
    - Clones or pulls the latest state into a local temp directory
    - Applies specified glob patterns to locate files of interest
    - Loads PDF files using `PDFLoader`
    - Loads general-purpose text files using `TextLoader`
    - Adds chunk metadata including `chunk_id` and `chunk_total`
    - Returns a list of LangChain `Document` objects (does not push to vector DB)

    Attributes:
        config (Config): The application configuration, including:
            - temp_dir: base temp folder for local repo checkouts
            - repo_sources: list of dicts with {repo, globs}
            - chunk_size / chunk_overlap: passed through to loaders

    Example:
        >>> from config import Config
        >>> loader = GitLoader(Config.load())
        >>> docs = loader.load()
        >>> print(docs[0].metadata)
        {'source': 'docs/content/learn/getting-started.adoc', 'chunk_id': 0, 'chunk_total': 3}
    """

    def __init__(self, config: Config):
        """
        Initialize the GitLoader with a configuration.

        Args:
            config (Config): Configuration object with glob patterns, temp directory,
                             and chunking parameters used by sub-loaders.
        """
        self.config = config
        self.base_path = Path(config.temp_dir) / "repo_sources"
        self.pdf_loader = PDFLoader(config)
        self.text_loader = TextLoader(config)

    def load(self) -> List[Document]:
        """
        Load and chunk documents from all configured Git repositories.

        Process:
            1. Clone or update each repo into a local temp folder
            2. Match files based on repo-specific glob patterns
            3. Route files to PDF or text loader based on extension
            4. Chunk and annotate documents with source + chunk metadata

        Returns:
            List[Document]: A list of chunked LangChain documents ready for embedding or indexing.
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
        """
        Ensure a local copy of the Git repo is present and up to date.

        If the repo exists, attempts a `git pull`. If it fails or the repo
        is missing, performs a fresh `git clone`.

        Args:
            url (str): Git repository URL.
            dest (Path): Local path where the repo should reside.
        """
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
        """
        Clone the given Git repository to the destination directory.

        Args:
            url (str): Git repository URL.
            dest (Path): Target path for clone.

        Raises:
            RuntimeError: If the clone operation fails.
        """
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
        """
        Apply glob patterns to collect file paths in a repo directory.

        Args:
            base (Path): Root path of the repo checkout.
            patterns (List[str]): List of glob patterns (e.g. "docs/**/*.md").

        Returns:
            List[Path]: All matched paths under the repo root.
        """
        matched: List[Path] = []

        for pattern in patterns:
            found = list(base.glob(pattern))
            if not found:
                logger.warning("No files matched pattern '%s' in %s", pattern, base)
            matched.extend(found)

        logger.info("Matched %d files in %s", len(matched), base)
        return matched
