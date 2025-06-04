FROM registry.access.redhat.com/ubi9/python-312:9.5

USER root
WORKDIR /app

RUN dnf install -y \
    unixODBC \
    unixODBC-devel && \
    curl -sSL https://packages.microsoft.com/config/rhel/9/prod.repo -o /etc/yum.repos.d/mssql-release.repo && \
    ACCEPT_EULA=Y dnf install -y msodbcsql18 && \
    dnf clean all

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

COPY vector_db ./vector_db
COPY loaders ./loaders
COPY embed_documents.py .
COPY config.py .
COPY .env .

RUN chown -R 1001:0 .

USER 1001

CMD ./embed_documents.py
