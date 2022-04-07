FROM nvidia/cuda:10.1-runtime-ubuntu18.04

RUN apt-get update && apt-get install -y \
    tmux \
    ffmpeg libsm6 libxext6 \
    unzip tar \
    wget vim \
    python3-pip \

COPY requirements.txt ./
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

WORKDIR /home/Fre-GAN