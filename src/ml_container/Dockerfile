FROM nvcr.io/nvidia/l4t-pytorch:r32.7.1-pth1.10-py3

WORKDIR /app

RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 16FAAD7AF99A65E2
# Remove Kitware repository and install GStreamer packages
RUN rm -f /etc/apt/sources.list.d/kitware*.list && \
    apt-get update && apt-get install -y \
    python3-gi \
    python3-gst-1.0 \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-tools \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Add CUDA library links
RUN mkdir -p /usr/lib/aarch64-linux-gnu && \
    ln -sf /usr/local/cuda-10.2/targets/aarch64-linux/lib/libcurand.so.10 /usr/lib/aarch64-linux-gnu/libcurand.so.10 && \
    ln -sf /usr/local/cuda-10.2/targets/aarch64-linux/lib/libcublas.so.10 /usr/lib/aarch64-linux-gnu/libcublas.so.10

COPY requirements.txt .
RUN pip3 install -r requirements.txt
COPY . .

# CMD ["python3", "ml_models.py"]
