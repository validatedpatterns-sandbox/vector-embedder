# 📚 vector-embedder

[![Quay Repository](https://img.shields.io/badge/Quay.io-vector--embedder-blue?logo=quay)](https://quay.io/repository/hybridcloudpatterns/vector-embedder)
[![CI](https://github.com/validatedpatterns-sandbox/vector-embedder/actions/workflows/ci-pipeline.yaml/badge.svg?branch=main)](https://github.com/validatedpatterns-sandbox/vector-embedder/actions/workflows/ci-pipeline.yaml)


**vector-embedder** is a flexible, language-agnostic document ingestion and embedding pipeline. It transforms structured and unstructured content from multiple sources into vector embeddings and stores them in your vector database of choice.

It supports Git repositories, web URLs, and file types like Markdown, PDFs, and HTML. Designed for local runs, containers, or OpenShift/Kubernetes jobs.

- [📚 vector-embedder](#-vector-embedder)
  - [⚙️ Features](#️-features)
  - [🚀 Quick Start](#-quick-start)
    - [1. Configuration](#1-configuration)
    - [2. Run Locally](#2-run-locally)
    - [3. Or Run in a Container](#3-or-run-in-a-container)
  - [🧪 Dry Run Mode](#-dry-run-mode)
  - [📦 Dependency Management \& Updates](#-dependency-management--updates)
    - [🔧 Installing `pip-tools`](#-installing-pip-tools)
    - [➕ Adding / Updating a Package](#-adding--updating-a-package)
  - [🗂️ Project Layout](#️-project-layout)
  - [🧪 Local DB Testing](#-local-db-testing)
    - [PGVector (PostgreSQL)](#pgvector-postgresql)
    - [Elasticsearch](#elasticsearch)
    - [Redis (RediSearch)](#redis-redisearch)
    - [Qdrant](#qdrant)
    - [SQL Server (MSSQL)](#sql-server-mssql)
  - [🙌 Acknowledgments](#-acknowledgments)

---

## ⚙️ Features

- ✅ **Multi-DB support**:
  - Redis (RediSearch)
  - Elasticsearch
  - PGVector (PostgreSQL)
  - SQL Server (preview)
  - Qdrant
  - Dry Run (no DB required; logs to console)
- ✅ **Flexible input sources**:
  - Git repositories via glob patterns (`**/*.pdf`, `*.md`, etc.)
  - Web pages via configurable URL lists
- ✅ **Smart chunking** with configurable `CHUNK_SIZE` and `CHUNK_OVERLAP`
- ✅ Embeddings via [`sentence-transformers`](https://www.sbert.net/)
- ✅ Parsing via [LangChain](https://github.com/langchain-ai/langchain) + [Unstructured](https://unstructured.io/)
- ✅ UBI-compatible container, OpenShift-ready
- ✅ Fully configurable via `.env` or `-e` environment flags

---

## 🚀 Quick Start

### 1. Configuration

Set your configuration in a `.env` file at the project root.

```dotenv
# Temporary working directory
TEMP_DIR=/tmp

# Logging
LOG_LEVEL=info

# Sources
REPO_SOURCES=[{"repo": "https://github.com/example/repo.git", "globs": ["docs/**/*.md"]}]
WEB_SOURCES=["https://example.com/docs/", "https://example.com/report.pdf"]

# Chunking
CHUNK_SIZE=2048
CHUNK_OVERLAP=200

# Embeddings
EMBEDDING_MODEL=sentence-transformers/all-mpnet-base-v2

# Vector DB
DB_TYPE=DRYRUN
```

🧪 `DB_TYPE=DRYRUN` logs chunks to stdout and skips database indexing—great for development!

### 2. Run Locally

```bash
./embed_documents.py
```

### 3. Or Run in a Container

```bash
podman build -t embed-job .

podman run --rm --env-file .env embed-job
```

You can also pass inline vars:

```bash
podman run --rm \
  -e DB_TYPE=REDIS \
  -e REDIS_URL=redis://localhost:6379 \
  embed-job
```

---

## 🧪 Dry Run Mode

Dry run skips vector DB upload and prints chunk metadata and content to the terminal.

```dotenv
DB_TYPE=DRYRUN
```

Run it:

```bash
./embed_documents.py
```

---

## 📦 Dependency Management & Updates

This project keeps *two* dependency files under version control:

| File | Purpose | Edited by |
|------|---------|-----------|
| **`requirements.in`** | Short, human-readable list of *top-level* libraries (no pins) | You |
| **`requirements.txt`** | Fully-resolved, **pinned** lock file—including hashes—for exact, reproducible builds | `pip-compile` |

### 🔧 Installing `pip-tools`

```bash
python -m pip install --upgrade pip-tools
````

### ➕ Adding / Updating a Package

1. **Edit `requirements.in`**

   ```diff
   - sentence-transformers
   + sentence-transformers>=4.1
   + llama-index
   ```
2. **Re-lock** the environment

   ```bash
   pip-compile --upgrade
   ```
3. **Synchronise** your virtual-env

   ```bash
   pip-sync
   ```

---

## 🗂️ Project Layout

```
.
├── embed_documents.py      # Main entrypoint script
├── config.py               # Config loader from env
├── loaders/                # Git, web, PDF, and text loaders
├── vector_db/              # Pluggable DB providers
├── requirements.txt        # Python dependencies
├── redis_schema.yaml       # Redis index schema (if used)
└── .env                    # Default runtime config
```

---

## 🧪 Local DB Testing

Run a compatible DB locally to test full ingestion + indexing.

### PGVector (PostgreSQL)

```bash
podman run --rm -d \
  --name pgvector \
  -e POSTGRES_USER=user \
  -e POSTGRES_PASSWORD=pass \
  -e POSTGRES_DB=mydb \
  -p 5432:5432 \
  docker.io/ankane/pgvector
```

```bash
DB_TYPE=PGVECTOR ./embed_documents.py
```

---

### Elasticsearch

```bash
podman run --rm -d \
  --name elasticsearch \
  -p 9200:9200 \
  -e "discovery.type=single-node" \
  -e "xpack.security.enabled=true" \
  -e "ELASTIC_PASSWORD=changeme" \
  -e "ES_JAVA_OPTS=-Xms512m -Xmx512m" \
  docker.io/elastic/elasticsearch:8.11.1
```

```bash
DB_TYPE=ELASTIC ./embed_documents.py
```

---

### Redis (RediSearch)

```bash
podman run --rm -d \
  --name redis-stack \
  -p 6379:6379 \
  docker.io/redis/redis-stack-server:6.2.6-v19
```

```bash
DB_TYPE=REDIS ./embed_documents.py
```

---

### Qdrant

```bash
podman run -d \
  -p 6333:6333 \
  --name qdrant \
  docker.io/qdrant/qdrant
```

```bash
DB_TYPE=QDRANT ./embed_documents.py
```

---

### SQL Server (MSSQL)


```bash
podman run --rm -d \
  --name mssql \
  -e ACCEPT_EULA=Y \
  -e SA_PASSWORD=StrongPassword! \
  -p 1433:1433 \
  mcr.microsoft.com/mssql/rhel/server:2025-latest
````

```bash
DB_TYPE=MSSQL ./embed_documents.py
```

---

## 🙌 Acknowledgments

Built with:

- [LangChain](https://github.com/langchain-ai/langchain)
- [Unstructured](https://github.com/Unstructured-IO/unstructured)
- [Sentence Transformers](https://www.sbert.net/)
- [OpenShift UBI Base](https://catalog.redhat.com/software/containers/search)
