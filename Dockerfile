#
# Album Coverart Similarity Search
# @author Loreto Parisi (loretoparisi at gmail dot com)
# Copyright (c) 2020 Loreto Parisi (loretoparisi at gmail dot com)
#

FROM python:3.7.4-slim-buster

MAINTAINER Loreto Parisi loreto@musixmatch.com

WORKDIR app

COPY . .

# install pytorch, tensorflow from local archive
RUN pip install wheels/tensorflow-2.2.0-cp37-cp37m-manylinux2010_x86_64.whl && \
    pip install wheels/torch-1.5.0-cp37-cp37m-manylinux1_x86_64.whl

COPY requirements.txt /tmp/requirements.txt
RUN cat /tmp/requirements.txt | xargs -n 1 -L 1 pip3 install --no-cache-dir

CMD ["bash"]