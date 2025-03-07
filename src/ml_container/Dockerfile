# Use a lightweight Python base image
# FROM nvcr.io/nvidia/l4t-base:r32.7.1  # Jetson TX2 Base Image (Python 3.6 included) todo: uncomment this out and comment out the linux version
FROM python:3.8-slim

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Run ML model script on container start
CMD ["python", "-u", "ml_models.py"]
