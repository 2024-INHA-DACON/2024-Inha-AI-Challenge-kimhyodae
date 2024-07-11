# 2024-Inha-AI-Challenge
2024 Inha AI Challenge

master 브랜치 사용중

# requirements.txt
transformers

# Ref.
https://teddylee777.github.io/langchain/rag-tutorial//

# Dockerfile
# Use an official Ubuntu as a parent image
FROM ubuntu:20.04

# Set environment variables to non-interactive to avoid prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install necessary utilities
RUN apt-get update && apt-get install -y \
    wget \
    bzip2 \
    curl \
    ca-certificates \
    sudo \
    git \
    && rm -rf /var/lib/apt/lists/*

# RUN apt-get update && apt-get install -y python3-pip


# Install Miniconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    /bin/bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh

# Set path to conda
ENV PATH=/opt/conda/bin:$PATH

# Create a conda environment and install PyTorch and other dependencies
RUN conda create -y --name pytorch_env python=3.8 && \
    /bin/bash -c "source activate pytorch_env" && \
    conda install -y -n pytorch_env pytorch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 pytorch-cuda=11.8 -c pytorch -c nvidia

# Set the default environment to pytorch_env
ENV CONDA_DEFAULT_ENV=pytorch_env
ENV PATH=/opt/conda/envs/pytorch_env/bin:$PATH

# Ensure the environment is activated
RUN echo "source activate pytorch_env" >> ~/.bashrc

# Set the working directory
WORKDIR /home/Chatbot/FastAPI

# Copy the current directory contents into the container at /workspace
COPY ./requirements.txt /home/Chatbot/FastAPI

# Install any remaining dependencies from a requirements file, if needed
RUN pip install -r requirements.txt
