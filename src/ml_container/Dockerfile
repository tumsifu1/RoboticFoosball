# Use NVIDIA PyTorch container
FROM nvcr.io/nvidia/l4t-pytorch:r32.7.1-pth1.10-py3

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip3 install -r requirements.txt

# Copy source code
COPY . .

# Run ML script
CMD ["python3", "ml_models.py"]
