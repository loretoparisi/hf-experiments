#
# hf-experiments
# @author Loreto Parisi (loretoparisi at gmail dot com)
# Copyright (c) 2020-2021 Loreto Parisi (loretoparisi at gmail dot com)
#

FROM python:3.7.4-slim-buster

LABEL maintainer Loreto Parisi loretoparisi@gmail.com

WORKDIR app

COPY src .

# system-wide dependencies
RUN apt-get update && apt-get install -y \
    software-properties-common \
    libsndfile1-dev \
    curl && \
    add-apt-repository ppa:jonathonf/ffmpeg-4 && \
    apt-get install -y ffmpeg

# system-wide python requriments
RUN pip3 install -r requirements-dev.txt

# utils
RUN pip3 install -r lpdutils/requirements.txt

# experiment-wide python requriments
RUN pip3 install -r asr/requirements.txt
RUN pip3 install -r genre/requirements.txt
RUN pip3 install -r audioset/requirements.txt
RUN pip3 install -r audioseg/requirements.txt
RUN pip3 install -r mlpvision/requirements.txt
RUN pip3 install -r skweak/requirements.txt

CMD ["bash"]