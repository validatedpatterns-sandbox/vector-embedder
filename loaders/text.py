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
        """Partition → group → (optional) secondary split → add chunk indices."""
        grouped: list[Document] = []

        for path in paths:
            try:
                logger.info("Partitioning %s", path)
                elements = partition(filename=str(path), strategy="fast")

                buf, buf_len, chunk_idx = [], 0, 0
                fname = Path(path).name

                def _flush():
                    nonlocal buf, buf_len, chunk_idx
                    if not buf_len:
                        return
                    grouped.append(
                        Document(
                            page_content="\n".join(buf).strip(),
                            metadata={
                                "source": str(path),
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
                        buf.append(f"## {fname}\n")  # one heading per chunk
                    if buf_len + len(txt) > self.config.chunk_size:
                        _flush()
                    buf.append(txt)
                    buf_len += len(txt)
                _flush()

            except Exception as e:
                logger.warning("Failed to load %s: %s", path, e)

        # — optional secondary split for ultra‑long groups —
        final_docs = []
        for doc in grouped:
            if len(doc.page_content) > self.config.chunk_size * 2:
                final_docs.extend(self.splitter.split_documents([doc]))
            else:
                final_docs.append(doc)

        # annotate chunk_total (needed only once per file)
        counts: dict[str, int] = {}
        for d in final_docs:
            counts[d.metadata["source"]] = counts.get(d.metadata["source"], 0) + 1
        for d in final_docs:
            d.metadata["chunk_total"] = counts[d.metadata["source"]]

        logger.info(
            "Produced %d chunks (avg %.0f chars)",
            len(final_docs),
            sum(len(d.page_content) for d in final_docs) / max(1, len(final_docs)),
        )
        return final_docs
