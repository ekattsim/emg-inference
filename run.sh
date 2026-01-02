#!/usr/bin/env bash

IMAGE_NAME="emg-inference-env"

# Build the image using the Dockerfile in the current directory
echo "Building/Updating Docker image..."
sudo docker build -t $IMAGE_NAME .

# Run the container with required hardware flags
echo "Starting container..."
sudo docker run \
    --gpus all \
    --runtime=nvidia \
    --privileged \
    --network host \
    -v /var/run/dbus:/var/run/dbus \
    --rm -it \
    -v "$(pwd)":/workspace \
    $IMAGE_NAME
