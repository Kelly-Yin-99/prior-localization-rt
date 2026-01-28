# syntax=docker/dockerfile:1
FROM continuumio/miniconda3:24.7.1-0

SHELL ["/bin/bash", "-lc"]

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create conda environment
ARG PYTHON_VERSION=3.10
RUN conda create -n priorloc python=${PYTHON_VERSION} pip -y && \
    conda clean -afy

# Activate environment by default
ENV PATH=/opt/conda/envs/priorloc/bin:$PATH
ENV CONDA_DEFAULT_ENV=priorloc

# This is the "cd" inside the container
WORKDIR /opt/prior-localization

# Copy code from the directory you ran `docker build` in
COPY . .

# Install package
RUN pip install -U pip && pip install -e .

# Default test command
CMD ["python", "-c", "import prior_localization; print('prior_localization OK')"]

