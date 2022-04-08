FROM pytorch/1.10.0-cuda11.3-cudnn8-runtime

RUN apt-get update && apt-get -y upgrade && apt-get install -y python-pip

RUN apt-get install -y python3-pip \
                       vim \
                       tmux \
                       ffmpeg \
                       libsm6 \
                       libxext6 \
                       unzip \
                       tar \
                       curl \
                       wget

COPY requirements.txt /root/Fre-GAN/requirements.txt
WORKDIR /root/Fre-GAN

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY . /root/Fre-GAN

ENV PYTHONPATH "${PYTHONPATH}:/home/Fre-GAN"