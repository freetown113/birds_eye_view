#!/bin/bash

# Get location of this script to orginize the distribution of files
location=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# Set segmentation project directory name
name="mmsegmentation"

# Download project that contains semantic segmentation algorithm
git clone https://github.com/open-mmlab/mmsegmentation.git $location/$name && \

# Download appropriate weights for the segmentation model and locate in the convinient directory
mkdir $location/$name/models && \
wget https://download.openmmlab.com/mmsegmentation/v0.5/pspnet/pspnet_r101-d8_512x512_160k_ade20k/pspnet_r101-d8_512x512_160k_ade20k_20200615_100650-967c316f.pth -P $location/$name/models && \

# Create source file in order to update environmental variables
echo "export PYTHONPATH=:$location/$name" > source.bash
