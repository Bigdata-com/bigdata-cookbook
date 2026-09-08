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

ENV PIP_ROOT_USER_ACTION=ignore

# Build context must be the cookbook root (see README).
# Installs Thematic_Screener_CLI pipeline deps; notebook delegates to that package.
COPY Thematic_Screener_CLI/requirements.txt /tmp/cli-requirements.txt
COPY Thematic_Screener/requirements.txt /tmp/thematic-requirements.txt

RUN python -m pip install --upgrade pip \
  && python -m pip install --no-cache-dir -r /tmp/cli-requirements.txt \
  && python -m pip install --no-cache-dir jupyterlab matplotlib

# Switch to non-root user
USER bigdata
WORKDIR /cookbook/Thematic_Screener

# Expose JupyterLab port
EXPOSE 8888

# Set the default command to start JupyterLab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
