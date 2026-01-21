FROM nvidia/cuda:12.9.1-cudnn-devel-ubuntu22.04

WORKDIR /workspace/piper

# Install system deps and NVSHMEM (Ray dependency)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-venv python3-pip \
    git ca-certificates \
    libnvshmem3-cuda-12 \
 && rm -rf /var/lib/apt/lists/*

# Make NVSHMEM visible to dynamic linker
RUN echo "/usr/lib/x86_64-linux-gnu/nvshmem/12" > /etc/ld.so.conf.d/nvshmem.conf && ldconfig

RUN python3 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Python tooling and remaining dependencies that aren't Ray or Torch
# (we mount and use the patched Ray and Torch installs from our .venv/lib/python3.10/site-packages directory instead)
RUN pip install --upgrade pip setuptools wheel

RUN pip install nvidia-cusparselt-cu12

RUN pip install \
    numpy \
    cupy-cuda12x \
    msgpack \
    grpcio \
    protobuf \
    filelock \
    requests \
    pyyaml \
    jsonschema \
    packaging

CMD ["/bin/bash"]