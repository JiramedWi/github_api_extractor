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

# Download and install Anaconda
RUN wget -O /tmp/Anaconda3.sh https://repo.anaconda.com/archive/Anaconda3-2023.07-2-Linux-x86_64.sh && \
    bash /tmp/Anaconda3.sh -b -p /opt/anaconda3 && \
    rm /tmp/Anaconda3.sh

# Set Anaconda environment variables
ENV PATH="/opt/anaconda3/bin:$PATH"

# Verify Anaconda installation
RUN conda --version

# Create a user "pee" inside the container
RUN useradd -m pee

# Set up the project directory under /home/pee/repo/github_api_extractor
RUN mkdir -p /home/pee/repo/github_api_extractor && \
    chown -R pee:pee /home/pee/repo

# Set working directory for the user
WORKDIR /home/pee/repo/github_api_extractor

# Switch to user "pee"
USER pee

# Copy the project into the container (as user pee)
COPY --chown=pee:pee . /home/pee/repo/github_api_extractor

# Set default shell to bash (so Conda activates properly)
SHELL ["/bin/bash", "-c"]

# Activate Conda environment and start a shell
CMD ["bash", "-c", "source activate ERAWAN_env && /bin/bash"]
