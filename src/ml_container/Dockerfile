# Use a lightweight Python base image
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
