FROM rayproject/ray:2.44.1-py310-cu128

USER root
RUN apt-get update && apt-get install -y --no-install-recommends graphviz && \
    rm -rf /var/lib/apt/lists/*
USER ray

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu128
