
ARG UBUNTU_VERSION=22.04
ARG NVIDIA_CUDA_VERSION=12.5.1

FROM ubuntu:${UBUNTU_VERSION} AS base

ARG UNAME=morita
ARG PASS=morita
ARG USER_UID=1000
ARG USER_GID=1000

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get upgrade -y

RUN apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    wget \
    python3-pip \
    python3-dev \
    python3-dev \
    openssh-server \
    vim \
    sudo \
    lsb-release 

RUN groupadd -g ${USER_GID} ${UNAME} && \
    useradd -m -s /bin/bash -u ${USER_UID} -g ${USER_GID} -G sudo ${UNAME} && \
    usermod -aG sudo ${UNAME} && \
    echo ${UNAME}:${PASS} | chpasswd && \
    echo "$UNAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

RUN apt-get install -y --no-install-recommends \
    cmake \
    git \
    ninja-build \
    libgtk2.0-dev \
    pkg-config \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    python3-pip \
    python3-dev \
    python3-numpy \
    libtbb2 \
    libtbb-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev

WORKDIR /home/${UNAME}
RUN git clone https://github.com/opencv/opencv.git && \
    git clone https://github.com/opencv/opencv_contrib.git && \
    cd opencv && \
    git checkout 4.5.3 && \
    cd ../opencv_contrib && \
    git checkout 4.5.3 && \
    cd ../opencv

WORKDIR /home/${UNAME}/opencv/build
RUN cmake -D CMAKE_BUILD_TYPE=Release \
          -D CMAKE_INSTALL_PREFIX=/usr/local \
          -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
          -D BUILD_EXAMPLES=ON .. \
          -GNinja && \
    ninja && \
    ninja install


USER ${UNAME}

CMD ["/bin/bash"]

