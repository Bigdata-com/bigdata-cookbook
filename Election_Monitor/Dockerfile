FROM python:3.13-alpine

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Install system dependencies for Python packages
RUN apk update && apk add --no-cache \
    curl \
    ca-certificates \
    build-base \
    libffi-dev \
    openssl-dev \
    sqlite-dev \
    gcc \
    musl-dev \
    git \
    graphviz

# Create non-root user 'bigdata' and necessary directories
RUN adduser --disabled-password bigdata 

# Remove setuid and setgid permissions for extra security
RUN find / -perm +6000 -type f -exec chmod a-s {} \; || true

# Copy requirements file before installing dependencies
COPY requirements.txt .

# Install Python dependencies && Install JupyterLab
ENV PIP_ROOT_USER_ACTION=ignore
RUN python -m pip install --upgrade pip \
  && python -m pip install --no-cache-dir -r requirements.txt \
  && python -m pip install --no-cache-dir jupyterlab

# Switch to non-root user
USER bigdata
WORKDIR /app

# Expose JupyterLab port
EXPOSE 8888

# Set the default command to start JupyterLab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"] 