import logging
from typing import List

import pyodbc
from langchain_core.documents import Document
from langchain_sqlserver import SQLServer_VectorStore

from vector_db.db_provider import DBProvider

logger = logging.getLogger(__name__)


class SQLServerProvider(DBProvider):
    """
    SQL Server-based vector DB provider using LangChain's SQLServer_VectorStore.

    Args:
        host (str): Hostname of the SQL Server
        port (str): Port number
        user (str): SQL login username
        password (str): SQL login password
        database (str): Database name to connect to or create
        table (str): Name of the table used to store vector embeddings
        driver (str): ODBC driver name (e.g., 'ODBC Driver 18 for SQL Server')

    Example:
        >>> provider = SQLServerProvider(
        ...     host="localhost",
        ...     port="1433",
        ...     user="sa",
        ...     password="StrongPassword!",
        ...     database="docs",
        ...     table="vector_table",
        ...     driver="ODBC Driver 18 for SQL Server"
        ... )
        >>> provider.add_documents(chunks)
    """

    def __init__(
        self,
        host: str,
        port: str,
        user: str,
        password: str,
        database: str,
        table: str,
        driver: str,
    ) -> None:
        super().__init__()

        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.database = database
        self.table = table
        self.driver = driver

        self.connection_string = self._build_connection_string(self.database)

        self._ensure_database_exists()
        self.db = SQLServer_VectorStore(
            connection_string=self.connection_string,
            embedding_function=self.embeddings,
            table_name=self.table,
            embedding_length=768,  # Should match the sentence-transformer model
        )

    def _build_connection_string(self, db_name: str) -> str:
        return (
            f"Driver={{{self.driver}}};"
            f"Server={self.host},{self.port};"
            f"Database={db_name};"
            f"UID={self.user};"
            f"PWD={self.password};"
            "TrustServerCertificate=yes;"
            "Encrypt=no;"
        )

    def _ensure_database_exists(self) -> None:
        master_conn_str = self._build_connection_string("master")
        try:
            with pyodbc.connect(master_conn_str, autocommit=True) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    f"IF DB_ID('{self.database}') IS NULL CREATE DATABASE [{self.database}]"
                )
                cursor.close()
        except Exception as e:
            logger.exception("Failed to ensure database '%s' exists", self.database)
            raise RuntimeError(
                f"Failed to ensure database '{self.database}' exists: {e}"
            )

    def add_documents(self, docs: List[Document]) -> None:
        """
        Add documents to the SQL Server table in batches.

        Args:
            docs (List[Document]): List of LangChain documents to embed and insert.

        Raises:
            Exception: If any batch insert fails.
        """
        batch_size = 50
        for i in range(0, len(docs), batch_size):
            batch = docs[i : i + batch_size]
            try:
                self.db.add_documents(batch)
            except Exception as e:
                logger.exception("Failed to insert batch starting at index %s", i)
                raise
