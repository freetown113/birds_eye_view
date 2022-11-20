FROM nvidia/cuda:11.6.0-cudnn8-devel-ubuntu20.04

#-------------------------------------------#
# Base installations
#-------------------------------------------#
ENV DEBIAN_FRONTEND noninteractive

# Install some basic utilities
RUN apt update && apt install -y \
    wget \
    nano \
    ca-certificates \
    git \
    openssh-server \
 && rm -rf /var/lib/apt/lists/*
 
#-------------------------------------------#
# Budgie installations
#-------------------------------------------#

ENV DEBIAN_FRONTEND noninteractive
ENV TZ 'Europe/Ljubljana'

RUN apt update && echo $TZ > /etc/timezone && \
    apt update && apt install -y tzdata \
    tree && \
    rm /etc/localtime && \
    ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && \
    dpkg-reconfigure -f noninteractive tzdata

RUN apt update &&\
    	apt install -y software-properties-common \
	build-essential \
	ffmpeg \ 
    libsm6 \
    libxext6 \
	python3 \
	python3-pip &&\
	apt-get clean &&\
	ln -s /usr/bin/python3 /usr/local/bin/python &&\
    	ln -s /usr/bin/python3 /usr/local/bin/python3 &&\    
	rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

#-------------------------------------------#
# Pytorch and other libraries
#-------------------------------------------#

# Install essential Python packages
RUN python3 -m pip --no-cache-dir install \
    torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html \
    mmcv-full==1.4.2 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html \
    timm==0.4.12 \
    mmdet==2.22.0 \
    mmsegmentation==0.20.2 \
    opencv-python \
    scipy
    