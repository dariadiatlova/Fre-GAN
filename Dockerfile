FROM nvidia/cuda:10.1-runtime-ubuntu18.04

RUN apt-get update && apt-get -y upgrade \
RUN apt-get install -y vim-common \
                       tmux \
                       ffmpeg \
                       libsm6 \
                       libxext6 \
                       unzip \
                       tar \
                       wget \
                       python3-pip \

COPY requirements.txt /root/requirements.txt
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

RUN mkdir /home/Fre-GAN
ADD . /home/Fre-GAN
WORKDIR /home/Fre-GAN

ENV PYTHONPATH "${PYTHONPATH}:/home/Fre-GAN"