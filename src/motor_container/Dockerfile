# FROM nvcr.io/nvidia/l4t-base:r32.7.1  # Jetson TX2 Base Image (Python 3.6 included) todo: uncomment this out and comment out the linux version
FROM python:3.6-slim-buster

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
CMD ["python3.6", "motor_control.py"]
