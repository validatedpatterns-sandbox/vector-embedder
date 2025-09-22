FROM registry.access.redhat.com/ubi10/python-312-minimal:10.0

USER root

WORKDIR /app

RUN microdnf install -y git \
    unixODBC \
    unixODBC-devel && \
    curl -sSL https://packages.microsoft.com/config/rhel/9/prod.repo -o /etc/yum.repos.d/mssql-release.repo && \
    ACCEPT_EULA=Y microdnf install -y msodbcsql18 && \
    microdnf clean all

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install \
      --no-cache-dir \
      --compile \
      -r requirements.txt

COPY vector_db ./vector_db
COPY loaders ./loaders
COPY embed_documents.py .
COPY config.py .
COPY .env .

RUN chown -R 1001:0 .

USER 1001

CMD ["python", "./embed_documents.py"]
