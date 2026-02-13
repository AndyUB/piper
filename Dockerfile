# Use a minimal Python base image
FROM python:3.10-slim

# Set working directory inside the container
WORKDIR /piper

RUN pip install numpy ray==2.53.0 cupy-cuda12x
RUN pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128


