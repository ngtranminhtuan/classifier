# Use Python 3.10
FROM python:3.10

# Working directory
WORKDIR /usr/src/app

# Copy requirements.txt to container before install dependencies
COPY requirements.txt ./requirements.txt

# Install dependencies
RUN apt-get update && apt-get install -y \
    git \
    libgl1-mesa-glx && \
    rm -rf /var/lib/apt/lists/* && \
    pip install --upgrade pip setuptools wheel \
    pip install --no-cache-dir -r requirements.txt

# Copy data
COPY checkbox_state_v2 ./checkbox_state_v2

# Copy script train và inference to container
COPY train.py ./train.py
COPY inference.py ./inference.py

# Default cmd
CMD ["bash"]
