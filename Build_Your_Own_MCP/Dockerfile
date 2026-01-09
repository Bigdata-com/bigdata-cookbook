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
RUN adduser --disabled-password --home /app bigdata && \
    mkdir /data && \
    chown bigdata:bigdata /app /data

# Remove setuid and setgid permissions for extra security
RUN find / -perm +6000 -type f -exec chmod a-s {} \; || true

# Copy uv binary from external image before switching to non-root user
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Add code
ADD build_your_mcp.py /app/build_your_mcp.py

# Switch to non-root user
USER bigdata
WORKDIR /app

# Expose MCP port
EXPOSE 8000

# Set the default command to start MCP
CMD ["uv", "run", "build_your_mcp.py"]