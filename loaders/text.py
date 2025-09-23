"""Text document loader for processing various text-based file formats."""

import logging
from pathlib import Path
from typing import List

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from unstructured.partition.auto import partition

from config import Config

logger = logging.getLogger(__name__)


class TextLoader:
    """
    Loads and semantically splits general-purpose text documents for RAG use.

    This loader uses Unstructured's local partitioning engine to break documents
    into semantically meaningful elements (e.g., titles, narrative text, lists),
    and then groups those elements into chunked LangChain Document objects,
    each annotated with `chunk_id` and `chunk_total` metadata for use in
    neighborhood-aware retrieval.

    This loader works fully offline and does not require an API key.

    Attributes:
        config (Config): Configuration object specifying chunking parameters.

    Example:
        >>> from pathlib import Path
        >>> from config import Config
        >>> loader = TextLoader(Config(chunk_size=1024, chunk_overlap=100))
        >>> docs = loader.load([Path("README.md"), Path("guide.adoc")])
        >>> print(docs[0].metadata)
        {'source': 'README.md', 'chunk_id': 0, 'chunk_total': 3}
    """

    def __init__(self, config: Config):
        """
        Initializes the TextLoader with chunking parameters.

        Args:
            config (Config): Application-level configuration object, must include:
                - config.chunk_size (int): Max number of characters per chunk.
                - config.chunk_overlap (int): Optional overlap between chunks.
        """
        self.config = config
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
        )

    def _process_single_file(self, path: Path) -> List[Document]:
        """Process a single file and return its document chunks."""
        logger.info("Partitioning %s", path)
        elements = partition(filename=str(path), strategy="fast")

        buf: List[str] = []
        buf_len, chunk_idx = 0, 0
        fname = path.name
        source_str = str(path)
        chunks = []

        def _flush():
            nonlocal buf, buf_len, chunk_idx
            if not buf_len:
                return
            chunks.append(
                Document(
                    page_content="\n".join(buf).strip(),
                    metadata={
                        "source": source_str,
                        "chunk_id": chunk_idx,
                    },
                )
            )
            buf, buf_len = [], 0
            chunk_idx += 1

        for el in elements:
            txt = getattr(el, "text", "").strip()
            if not txt:
                continue
            if buf_len == 0:
                buf.append(f"## {fname}\n")  # inject heading
            if buf_len + len(txt) > self.config.chunk_size:
                _flush()
            buf.append(txt)
            buf_len += len(txt)
        _flush()

        return chunks

    def _add_chunk_totals(self, docs: List[Document]) -> None:
        """Add chunk_total metadata to all documents."""
        counts: dict[str, int] = {}
        for doc in docs:
            source = doc.metadata["source"]
            counts[source] = counts.get(source, 0) + 1
        for doc in docs:
            doc.metadata["chunk_total"] = counts[doc.metadata["source"]]

    def load(self, paths: List[Path]) -> List[Document]:
        """
        Loads and splits a list of text files into semantic chunks.

        This function uses Unstructured's `partition()` function to extract
        structured document elements, then assembles these into grouped chunks
        (respecting max chunk size) while tagging each chunk with:

            - 'source': the original file path
            - 'chunk_id': the position of this chunk in the document
            - 'chunk_total': total number of chunks for that file

        Args:
            paths (List[Path]): List of file paths to text-based documents
                                (.txt, .md, .adoc, .rst, etc.).

        Returns:
            List[Document]: A list of LangChain `Document` objects, each
                            containing chunked text and structured metadata.

        Notes:
            - If a grouped chunk exceeds 2x chunk_size, it is re-split using
              a recursive character splitter.
            - Each chunk begins with a lightweight heading that includes the
              filename to help orient the LLM when formatting prompts.
        """
        grouped = []

        for path in paths:
            try:
                chunks = self._process_single_file(path)
                grouped.extend(chunks)
            except Exception as e:
                logger.warning("Failed to load %s: %s", path, e)

        # Handle oversized chunks via recursive splitting
        final_docs = []
        for doc in grouped:
            if len(doc.page_content) > self.config.chunk_size * 2:
                final_docs.extend(self.splitter.split_documents([doc]))
            else:
                final_docs.append(doc)

        # Add chunk_total metadata for all docs
        self._add_chunk_totals(final_docs)

        logger.info(
            "Produced %d chunks (avg %.0f chars)",
            len(final_docs),
            sum(len(d.page_content) for d in final_docs) / max(1, len(final_docs)),
        )
        return final_docs
