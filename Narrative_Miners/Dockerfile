# Use Python 3.11 slim image as base
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Create virtual environment and install dependencies
RUN python -m venv .venv && \
    . .venv/bin/activate && \
    pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install jupyterlab && \
    pip install python-dotenv

# Copy project files
COPY . .

# Create a non-root user for security
RUN useradd -m -u 1000 jupyter && \
    chown -R jupyter:jupyter /app
USER jupyter

# Activate virtual environment and set PATH
ENV PATH="/app/.venv/bin:$PATH"

# Expose JupyterLab port
EXPOSE 8888

# Set the default command to start JupyterLab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888"] 