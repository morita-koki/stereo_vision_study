#!/bin/bash

CONTAINER_NAME='stereo_vision_dev'
IMAGE_NAME='stereo_vision:latest'

# container is not running, start it
if [ ! "$(docker container ps -q -f name=$CONTAINER_NAME)" ]; then
    docker container start $CONTAINER_NAME
fi


docker exec -it $CONTAINER_NAME /bin/bash