FROM nvcr.io/nvidia/tensorflow:25.02-tf2-py3-igpu

# Install dependencies
RUN apt-get update && \
    apt-get install -y bluez && \
    rm -rf /var/lib/apt/lists/* && \
    pip install bleak mindrove

WORKDIR /workspace
