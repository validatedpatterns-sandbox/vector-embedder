import logging
import re
from typing import List, Optional

import pyodbc
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_sqlserver import SQLServer_VectorStore

from vector_db.db_provider import DBProvider

logger = logging.getLogger(__name__)


class MSSQLProvider(DBProvider):
    """
    SQL Server-based vector DB provider using LangChain's SQLServer_VectorStore integration.

    This provider connects to a Microsoft SQL Server instance using a full ODBC connection string,
    and stores document embeddings in a specified table. The target database will be created if it
    does not already exist.

    Attributes:
        db (SQLServer_VectorStore): Underlying LangChain-compatible vector store.
        connection_string (str): Full ODBC connection string to the SQL Server instance.

    Args:
        embeddings (HuggingFaceEmbeddings): Pre-initialized embeddings instance.
        connection_string (str): Full ODBC connection string (including target DB).
        table (str): Table name to store vector embeddings.

    Example:
        >>> from langchain_huggingface import HuggingFaceEmbeddings
        >>> from vector_db.mssql_provider import MSSQLProvider
        >>> embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")
        >>> provider = MSSQLProvider(
        ...     embeddings=embeddings,
        ...     connection_string="Driver={ODBC Driver 18 for SQL Server};Server=localhost,1433;Database=docs;UID=sa;PWD=StrongPassword!;TrustServerCertificate=yes;Encrypt=no;",
        ...     table="embedded_docs",
        ... )
        >>> provider.add_documents(docs)
    """

    def __init__(
        self,
        embeddings: HuggingFaceEmbeddings,
        connection_string: str,
        table: str,
    ) -> None:
        """
        Initialize the MSSQLProvider.

        Args:
            embeddings (HuggingFaceEmbeddings): HuggingFace-compatible embedding model instance.
            connection_string (str): Full ODBC connection string including target database name.
            table (str): Table name to store document embeddings.
        """
        super().__init__(embeddings)

        self.connection_string = connection_string
        self.table = table

        self._ensure_database_exists()

        server = self._extract_server_address()

        logger.info(
            "Connected to MSSQL instance at %s (table: %s)",
            server,
            self.table,
        )

        self.db = SQLServer_VectorStore(
            connection_string=self.connection_string,
            embedding_function=self.embeddings,
            table_name=self.table,
            embedding_length=self.embedding_length,
        )

    def _extract_server_address(self) -> str:
        match = re.search(r"Server=([^;]+)", self.connection_string, re.IGNORECASE)
        return match.group(1) if match else "unknown"

    def _extract_database_name(self) -> Optional[str]:
        match = re.search(r"Database=([^;]+)", self.connection_string, re.IGNORECASE)
        return match.group(1) if match else None

    def _build_connection_string_for_master(self) -> str:
        parts = self.connection_string.split(";")
        updated_parts = [
            "Database=master" if p.strip().lower().startswith("database=") else p
            for p in parts
            if p
        ]
        return ";".join(updated_parts) + ";"

    def _ensure_database_exists(self) -> None:
        database = self._extract_database_name()
        if not database:
            raise RuntimeError("No database name found in connection string.")

        master_conn_str = self._build_connection_string_for_master()
        try:
            with pyodbc.connect(master_conn_str, autocommit=True) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    f"IF DB_ID('{database}') IS NULL CREATE DATABASE [{database}]"
                )
                cursor.close()
        except pyodbc.ProgrammingError as e:
            if "1801" in str(e):
                logger.info("Database %s already exists, continuing", database)
                return
            logger.exception("Failed to ensure database '%s' exists", database)
            raise RuntimeError(
                f"Failed to ensure database '{database}' exists: {e}"
            ) from e

    def add_documents(self, docs: List[Document]) -> None:
        """
        Add documents to the SQL Server table in small batches.

        Args:
            docs (List[Document]): LangChain document chunks to embed and store.

        Raises:
            Exception: If a batch insert operation fails.
        """
        batch_size = 50
        for i in range(0, len(docs), batch_size):
            batch = docs[i : i + batch_size]
            try:
                self.db.add_documents(batch)
            except Exception:
                logger.exception("Failed to insert batch starting at index %s", i)
                raise
