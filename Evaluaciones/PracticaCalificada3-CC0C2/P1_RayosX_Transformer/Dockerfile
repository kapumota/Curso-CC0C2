# syntax=docker/dockerfile:1
FROM python:3.11-slim

# Evita prompts y reduce capa
ENV DEBIAN_FRONTEND=noninteractive     PIP_NO_CACHE_DIR=1     PYTHONDONTWRITEBYTECODE=1     PYTHONUNBUFFERED=1

# Dependencias del sistema mínimas (ajustables por estudiante si hace falta)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        curl \
        ca-certificates \
        tini \
        && rm -rf /var/lib/apt/lists/*

# Usuario no root
RUN useradd -ms /bin/bash appuser
WORKDIR /workspace
USER appuser

# Copiar requirements y preinstalar
COPY --chown=appuser:appuser requirements.txt /workspace/requirements.txt
RUN python -m pip install --upgrade pip && pip install -r requirements.txt

# Puerto típico para Jupyter
EXPOSE 8888

# Entrypoint seguro
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["bash", "-lc", "jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --NotebookApp.token='' --NotebookApp.password=''"]
