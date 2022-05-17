FROM ubuntu:20.04

USER root
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=en_US.UTF-8

RUN apt-get update && \
    apt-get install -y consul python3 python3-pip curl vim
RUN apt update
RUN apt-get update
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 30
RUN update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 30
RUN pip install --upgrade pip
RUN python -m pip install aiohttp protobuf grpcio cattrs awscli==1.22.19 awscli_plugin_endpoint

ARG WORK_DIR=/opt/script
RUN mkdir -pv "${WORK_DIR}"
ADD . $WORK_DIR
RUN chmod -R 777 ${WORK_DIR}

WORKDIR $WORK_DIR