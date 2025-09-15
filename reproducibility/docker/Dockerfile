# Dockerfile for Sentiment Feedback Analysis Reproduction
# Minimal image with Python + LaTeX for reproducible research

FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    # LaTeX (minimal installation)
    texlive-latex-base \
    texlive-latex-extra \
    texlive-fonts-recommended \
    texlive-science \
    latexmk \
    # Build tools
    build-essential \
    gcc \
    g++ \
    # Git for version control
    git \
    # Make for build system
    make \
    # Cleanup
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Copy requirements first for better caching
COPY reproducibility/requirements.txt /workspace/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . /workspace/

# Create necessary directories
RUN mkdir -p /workspace/project_files/build \
    /workspace/project_files/logs \
    /workspace/project_files/tables_figures/final_figures \
    /workspace/project_files/tables_figures/latex

# Set permissions for scripts
RUN chmod +x /workspace/reproducibility/run.sh

# Default command
CMD ["make", "all"]

# Build instructions:
# docker build -t sentiment-feedback .
# 
# Run instructions:
# docker run -v $(pwd)/project_files:/workspace/project_files sentiment-feedback
# 
# Interactive mode:
# docker run -it sentiment-feedback bash
#
# With custom make target:
# docker run sentiment-feedback make irf
