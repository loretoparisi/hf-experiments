#
# hf-experiments
# @author Loreto Parisi (loretoparisi at gmail dot com)
# Copyright (c) 2020-2022 Loreto Parisi (loretoparisi at gmail dot com)
#

FROM python:3.7.4-slim-buster

LABEL maintainer Loreto Parisi loretoparisi@gmail.com

WORKDIR app

COPY src .

# system-wide dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    software-properties-common \
    libsndfile1-dev \
    curl

# ffmpeg - https://johnvansickle.com/ffmpeg/
RUN curl https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-i686-static.tar.xz -O && \
    tar -xf ffmpeg-release-i686-static.tar.xz && \
    chmod 744 ffmpeg-5.1.1-i686-static/ffmpeg && \ 
    chmod 744 ffmpeg-5.1.1-i686-static/ffprobe && \ 
    mv ffmpeg-5.1.1-i686-static/ffmpeg /usr/local/bin && \
    mv ffmpeg-5.1.1-i686-static/ffprobe /usr/local/bin

# system-wide python requriments
RUN pip3 install -r requirements-dev.txt

# utils
RUN pip3 install -r lpdutils/requirements.txt

# experiment-wide python requriments
RUN pip3 install -r asr/requirements.txt
RUN pip3 install -r genre/requirements.txt
RUN pip3 install -r audioset/requirements.txt
#RUN pip3 install -r audioseg/requirements.txt
RUN pip3 install -r mlpvision/requirements.txt
RUN pip3 install -r skweak/requirements.txt
#RUN pip3 install -r pokemon/requirements.txt
RUN pip3 install -r projected_gan/requirements.txt
RUN pip3 install -r fasttext/requirements.txt
RUN pip3 install -r whisper/requirements.txt

CMD ["bash"]