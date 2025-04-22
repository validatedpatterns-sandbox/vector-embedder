import logging
from typing import List

import pyodbc
from langchain_core.documents import Document
from langchain_sqlserver import SQLServer_VectorStore

from vector_db.db_provider import DBProvider

logger = logging.getLogger(__name__)


class SQLServerProvider(DBProvider):
    """
    SQL Server-based vector DB provider using LangChain's SQLServer_VectorStore integration.

    This provider connects to a Microsoft SQL Server instance and stores document embeddings
    in a specified table. If the target database does not exist, it will be created automatically.

    Attributes:
        db (SQLServer_VectorStore): Underlying LangChain-compatible vector store.
        connection_string (str): Full ODBC connection string to the SQL Server instance.

    Args:
        embedding_model (str): HuggingFace-compatible embedding model to use.
        host (str): SQL Server hostname or IP address.
        port (str): Port number (typically 1433).
        user (str): SQL Server login username.
        password (str): SQL Server login password.
        database (str): Target database name. Will be created if not present.
        table (str): Table name to store vector embeddings.
        driver (str): ODBC driver name (e.g., 'ODBC Driver 18 for SQL Server').

    Example:
        >>> provider = SQLServerProvider(
        ...     embedding_model="BAAI/bge-large-en-v1.5",
        ...     host="localhost",
        ...     port="1433",
        ...     user="sa",
        ...     password="StrongPassword!",
        ...     database="my_vectors",
        ...     table="embedded_docs",
        ...     driver="ODBC Driver 18 for SQL Server"
        ... )
        >>> provider.add_documents(docs)
    """

    def __init__(
        self,
        embedding_model: str,
        host: str,
        port: str,
        user: str,
        password: str,
        database: str,
        table: str,
        driver: str,
    ) -> None:
        super().__init__(embedding_model)

        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.database = database
        self.table = table
        self.driver = driver

        self.connection_string = self._build_connection_string(self.database)
        self._ensure_database_exists()

        logger.info(
            "Connected to SQL Server at %s:%s, database: %s",
            self.host,
            self.port,
            self.database,
        )

        self.db = SQLServer_VectorStore(
            connection_string=self.connection_string,
            embedding_function=self.embeddings,
            table_name=self.table,
            embedding_length=768,  # Ensure this matches the model you're using
        )

    def _build_connection_string(self, db_name: str) -> str:
        """
        Construct a SQL Server ODBC connection string.

        Args:
            db_name (str): Name of the database to connect to.

        Returns:
            str: ODBC-compliant connection string.
        """
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
        """
        Connect to the SQL Server master database and create the target database if missing.

        Raises:
            RuntimeError: If the database cannot be created or accessed.
        """
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
