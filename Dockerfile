FROM nvidia/cuda:10.1-runtime-ubuntu18.04

RUN set -xe \
        && apt-get -y upgrade \
        && apt-get install -y python3-pip \
        && apt-get install -y vim-common \
        && apt-get install -y tmux \
        && apt-get install -y ffmpeg \
        && apt-get install -y libsm6 \
        && apt-get install -y libxext6 \
        && apt-get install -y unzip \
        && apt-get install -y tar \
        && apt-get install -y wget


COPY requirements.txt /root/Fre-GAN/requirements.txt
WORKDIR /root/Fre-GAN

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY . /home/Fre-GAN

ENV PYTHONPATH "${PYTHONPATH}:/home/Fre-GAN"