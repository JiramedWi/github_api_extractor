# Use Ubuntu as base image
FROM ubuntu:latest

# Set the author label
LABEL authors="pee"

# Set environment variables to avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Update package list and install required dependencies
RUN apt update && apt install -y \
    wget \
    curl \
    bzip2 \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Detect architecture and install the correct Conda distribution
RUN ARCH=$(dpkg --print-architecture) && \
    if [ "$ARCH" = "amd64" ]; then \
        wget -O /tmp/Anaconda3.sh https://repo.anaconda.com/archive/Anaconda3-2023.07-2-Linux-x86_64.sh && \
        bash /tmp/Anaconda3.sh -b -p /opt/anaconda3 && \
        rm /tmp/Anaconda3.sh && \
        echo 'export PATH="/opt/anaconda3/bin:$PATH"' >> ~/.bashrc; \
    elif [ "$ARCH" = "arm64" ] || [ "$ARCH" = "aarch64" ]; then \
        wget -O /tmp/Miniforge3.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh && \
        bash /tmp/Miniforge3.sh -b -p /opt/miniforge3 && \
        rm /tmp/Miniforge3.sh && \
        echo 'export PATH="/opt/miniforge3/bin:$PATH"' >> ~/.bashrc; \
    else \
        echo "Unsupported architecture: $ARCH" && exit 1; \
    fi

# Set Conda environment variables
ENV PATH="/opt/anaconda3/bin:/opt/miniforge3/bin:$PATH"

# Verify Conda installation
RUN conda --version

# Create a user "pee" inside the container
RUN useradd -m pee

# Set up the project directory
RUN mkdir -p /home/pee/repo/github_api_extractor && \
    chown -R pee:pee /home/pee/repo

# Set working directory for the user
WORKDIR /home/pee/repo/github_api_extractor

# Switch to user "pee"
USER pee

# Copy the project into the container
COPY --chown=pee:pee . /home/pee/repo/github_api_extractor

# Install dependencies using ERAWAN_env.yml
RUN conda env create -f /home/pee/repo/github_api_extractor/ERAWAN_env.yml

# Activate Conda environment and set it as default
SHELL ["/bin/bash", "-c"]
RUN echo "conda activate ERAWAN_env" >> ~/.bashrc

# Start a bash shell
CMD ["bash"]
