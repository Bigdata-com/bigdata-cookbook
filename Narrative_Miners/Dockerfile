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
    git

# Create non-root user 'bigdata' and necessary directories
RUN adduser -D bigdata && \
    mkdir /app /data && \
    chown bigdata:bigdata /app /data

# Remove setuid and setgid permissions for extra security
RUN find / -perm +6000 -type f -exec chmod a-s {} \; || true

# Switch to non-root user
USER bigdata
WORKDIR /app

# Copy requirements file before installing dependencies
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Install JupyterLab
RUN pip install jupyterlab python-dotenv

# Copy project files
COPY . .

# Ensure JupyterLab is in the PATH (just in case)
ENV PATH="/home/bigdata/.local/bin:$PATH"

# Expose JupyterLab port
EXPOSE 8888

# Set the default command to start JupyterLab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"] 