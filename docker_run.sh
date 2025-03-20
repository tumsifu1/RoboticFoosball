#!/bin/bash
docker-compose up --build
docker-compose down
docker container prune -f
docker image prune -f