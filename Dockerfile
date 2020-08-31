FROM ubuntu:18.04
ENV DEBIAN_FRONTEND=teletype

COPY . /cluwords_preprocess
WORKDIR /cluwords_preprocess

# Update ubuntu and install python3.6
RUN apt-get update && \
    apt-get install -y --no-install-recommends apt-utils && \
    apt-get -yq install software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get -yq install python3.6 python3.6-dev python3-pip

RUN ln -sfn /usr/bin/python3.6 /usr/bin/python3 && \
    ln -sfn /usr/bin/python3 /usr/bin/python && \
    ln -sfn /usr/bin/pip3 /usr/bin/pip

# To beautifully print utf-8 characters
ENV PYTHONIOENCODING utf-8

RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt
RUN python3 -m spacy download en_core_web_sm
ENV LANG="C.UTF-8"
