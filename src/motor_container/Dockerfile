FROM python:3.8

WORKDIR /app

RUN apt-get update && apt-get install -y \
    python3 python3-pip \
    i2c-tools \
    python3-smbus \
    libgpiod2 \
    python3-libgpiod \
    libatlas-base-dev \
    python3-dev \
    python3-setuptools \
    cmake

# Install Jetson.GPIO
RUN pip3 install Jetson.GPIO

# Install dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt
RUN pip3 install --upgrade adafruit-blinka adafruit-platformdetect

# Copy source code
COPY . .

CMD ["python3", "motor_control.py"]

