FROM python:3.8

WORKDIR /app

RUN apt-get update && apt-get install -y i2c-tools python3-smbus libgpiod2
# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --upgrade adafruit-Blinka adafruit-platformdetect
# Copy source code
COPY . .

# Run motor control script
CMD ["python", "motor_control.py"]
