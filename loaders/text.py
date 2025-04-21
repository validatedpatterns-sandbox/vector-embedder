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
    Loads and semantically splits a list of general-purpose text documents
    (e.g., .txt, .md, .rst, .adoc) using Unstructured's local partitioning engine.

    This loader does not require an API key and works fully offline.

    Unstructured's `partition()` function breaks files into structured elements
    like titles, narrative text, lists, etc., which improves RAG chunk quality
    over naive fixed-size splitting.

    Attributes:
        config (Config): The job configuration, including chunk size and overlap.

    Example:
        >>> loader = TextLoader(config)
        >>> docs = loader.load([Path("README.md"), Path("guide.rst")])
        >>> print(docs[0].page_content)
    """

    def __init__(self, config: Config):
        self.config = config
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
        )

    def load(self, paths: List[Path]) -> List[Document]:
        """Load files → Unstructured elements → grouped chunks."""
        all_chunks: List[Document] = []

        for path in paths:
            try:
                logger.info("Partitioning %s", path)
                elements = partition(filename=str(path), strategy="fast")

                buf: List[str] = []
                seen: set[str] = set()
                buf_len = 0
                fname = Path(path).name

                for el in elements:
                    text = getattr(el, "text", "").strip()
                    if not text or text in seen:
                        continue  # skip blanks & exact duplicates
                    seen.add(text)

                    # If starting a new group, add a tiny heading once
                    if buf_len == 0:
                        buf.append(f"## {fname}\n")

                    # Flush if adding this element would exceed chunk_size
                    if buf_len + len(text) > self.config.chunk_size and buf:
                        all_chunks.append(
                            Document(
                                page_content="\n".join(buf).strip(),
                                metadata={"source": str(path)},
                            )
                        )
                        buf, seen, buf_len = [f"## {fname}\n"], set(), 0

                    buf.append(text)
                    buf_len += len(text)

                # — flush remainder —
                if buf_len:
                    all_chunks.append(
                        Document(
                            page_content="\n".join(buf).strip(),
                            metadata={"source": str(path)},
                        )
                    )

            except Exception as e:
                logger.warning("Failed to load %s: %s", path, e)

        # 2) secondary split for *very* large groups
        final_docs: List[Document] = []
        for doc in all_chunks:
            if len(doc.page_content) > self.config.chunk_size * 2:
                final_docs.extend(self.splitter.split_documents([doc]))
            else:
                final_docs.append(doc)

        avg = sum(len(d.page_content) for d in final_docs) / max(1, len(final_docs))
        logger.info("Produced %d chunks (avg %.0f chars)", len(final_docs), avg)
        return final_docs
