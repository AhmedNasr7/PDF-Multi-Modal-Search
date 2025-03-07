# Use Ubuntu with CUDA 12.4
FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

# Set environment variables to avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive \
    TZ=UTC

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 python3.11-venv python3.11-dev \
    git curl wget nano unzip \
    && rm -rf /var/lib/apt/lists/*

# Create a working directory
WORKDIR /app

# Set Python 3.11 as default
RUN ln -sf /usr/bin/python3.11 /usr/bin/python3
RUN ln -sf /usr/bin/python3.11 /usr/bin/python

# Install pip and upgrade it
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

# Copy project files
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose ports for both the API and Streamlit
EXPOSE 8000 8501

# Run both processes (server.py + app.py) in parallel
CMD bash -c "python server.py & streamlit run app.py --server.port=8501 --server.address=0.0.0.0"
