#!/bin/bash
docker build -t $1 .
docker run -it --runtime=nvidia -p 5000:5000/udp -p 5555:5555 -p 5556:5556 -e NVIDIA_VISIBLE_DEVICES=all -v /tmp:/tmp -v /usr/local/cuda-10.2:/usr/local/cuda-10.2:ro --ipc=host $1
docker container prune -f
docker image prune -f

