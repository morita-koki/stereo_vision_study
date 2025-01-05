#!/bin/bash

# create a new container
CONTAINER_NAME='stereo_vision_dev'
IMAGE_NAME='stereo_vision:latest'

docker container create \
        -v /media:/media \
        -v $HOME/workspace/stereo_vision:/home/$USER/stereo_vision \
        -v /tmp/.X11-unix:/tmp/.X11-unix \
        -v $HOME/.ssh:/home/$USER/.ssh \
        -e DISPLAY=$DISPLAY \
        -e NVIDIA_DISABLE_REQUIRE=true \
        -u $(id -u):$(id -g) \
        -w /home/$USER/stereo_vision \
        --gpus all \
        --privileged \
        -it \
        --name $CONTAINER_NAME \
        $IMAGE_NAME 

