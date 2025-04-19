# vector-embedder

[![Docker Repository on Quay](https://quay.io/repository/dminnear/vector-embedder/status "Docker Repository on Quay")](https://quay.io/repository/dminnear/vector-embedder)

**vector-embedder** is a flexible, language-agnostic document ingestion pipeline that generates and stores vector embeddings from structured and unstructured content.

It supports embedding content from Git repositories (via glob patterns), web URLs, and various file types into multiple vector database backends. It runs locally, in containers, or as a Kubernetes/OpenShift job.

---

## 📦 Features

- ✅ **Multiple vector DB backends supported**:
  - Redis (RediSearch)
  - Elasticsearch
  - PGVector (PostgreSQL)
  - SQL Server (preview)
  - Qdrant
  - Dry Run (prints to console, no DB required)
- ✅ **Flexible input sources**:
  - Git repositories via glob patterns (`**/*.pdf`, `*.md`, etc.)
  - Web pages via configurable URL lists
- ✅ **Smart document chunking** with configurable `CHUNK_SIZE` and `CHUNK_OVERLAP`
- ✅ Embedding powered by [`sentence-transformers`](https://www.sbert.net/)
- ✅ Parsing powered by LangChain and [Unstructured](https://unstructured.io/)
- ✅ Fully configurable via `.env` or runtime env vars
- ✅ Containerized using UBI and OpenShift-compatible images

---

## 🚀 Usage

### Configuration

All settings are read from a `.env` file at the project root. You can override values using `export` or `-e` flags in containers.

Example `.env`:

```dotenv
# === File System Config ===
TEMP_DIR=/tmp

# === Logging ===
LOG_LEVEL=info

# === Git Repo Document Sources ===
REPO_SOURCES=[{"repo": "https://github.com/RHEcosystemAppEng/llm-on-openshift.git", "globs": ["examples/notebooks/langchain/rhods-doc/*.pdf"]}]

# === Web Document Sources ===
WEB_SOURCES=["https://ai-on-openshift.io/getting-started/openshift/", "https://ai-on-openshift.io/getting-started/opendatahub/"]

# === Embedding Config ===
CHUNK_SIZE=1024
CHUNK_OVERLAP=40
DB_TYPE=DRY_RUN

# === Redis ===
REDIS_URL=redis://localhost:6379
REDIS_INDEX=docs
REDIS_SCHEMA=redis_schema.yaml

# === Elasticsearch ===
ELASTIC_URL=http://localhost:9200
ELASTIC_INDEX=docs
ELASTIC_USER=elastic
ELASTIC_PASSWORD=changeme

# === PGVector ===
PGVECTOR_URL=postgresql://user:pass@localhost:5432/mydb
PGVECTOR_COLLECTION_NAME=documents

# === SQL Server ===
SQLSERVER_HOST=localhost
SQLSERVER_PORT=1433
SQLSERVER_USER=sa
SQLSERVER_PASSWORD=StrongPassword!
SQLSERVER_DB=docs
SQLSERVER_TABLE=vector_table
SQLSERVER_DRIVER=ODBC Driver 18 for SQL Server

# === Qdrant ===
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=embedded_docs
```

> 💡 Default `DB_TYPE=DRY_RUN` skips DB upload and prints chunked docs to stdout — great for testing!

---

### 🔍 Dry Run Mode

Dry run mode helps you test loaders and document chunking without needing any database.

```dotenv
DB_TYPE=DRY_RUN
```

Dry run will:

- Load from web and Git sources
- Chunk content
- Print chunk metadata and contents to stdout

Run with:

```bash
./embed_documents.py
```

or inside a container:

```bash
podman run --rm --env-file .env embed-job
```

---

### 🛠️ Build the Container

```bash
podman build -t embed-job .
```

---

### 🧪 Run in a Container

With inline env vars:

```bash
podman run --rm \
  -e DB_TYPE=REDIS \
  -e REDIS_URL=redis://localhost:6379 \
  embed-job
```

Or using `.env`:

```bash
podman run --rm \
  --env-file .env \
  embed-job
```

In OpenShift or Kubernetes, mount the `.env` via `ConfigMap` or use `env` blocks.

---

## 📂 Project Structure

```
.
├── embed_documents.py      # Main entrypoint
├── config.py               # Loads config from .env
├── loaders/                # Git, web, PDF, and text file loaders
├── vector_db/              # DB provider implementations
├── requirements.txt        # Python dependencies
├── redis_schema.yaml       # Schema definition for Redis vector DB
└── .env                    # Default config (example provided)
```

---

## 🧪 Local Testing Backends

Use Podman to spin up local test databases for fast experimentation.

### 🐘 PGVector (PostgreSQL)

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

### 🔍 Elasticsearch

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

### 🧠 Redis (RediSearch)

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

### 🔮 Qdrant

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

## 🙏 Acknowledgments

- [LangChain](https://github.com/langchain-ai/langchain)
- [Unstructured](https://github.com/Unstructured-IO/unstructured)
- [Sentence Transformers](https://www.sbert.net/)
- [OpenShift UBI Base Images](https://catalog.redhat.com/software/containers/search)
