#
# Copyright 2022 DMetaSoul
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

FROM ubuntu:20.04

ENV LANG=en_US.UTF-8
ENV TZ=Asia/Shanghai
ENV DEBIAN_FRONTEND=noninteractive
RUN echo "deb http://repo.huaweicloud.com/ubuntu/ focal main restricted" >/etc/apt/sources.list
RUN echo "deb http://repo.huaweicloud.com/ubuntu/ focal-updates main restricted" >>/etc/apt/sources.list
RUN echo "deb http://repo.huaweicloud.com/ubuntu/ focal universe" >>/etc/apt/sources.list
RUN echo "deb http://repo.huaweicloud.com/ubuntu/ focal-updates universe" >>/etc/apt/sources.list
RUN echo "deb http://repo.huaweicloud.com/ubuntu/ focal multiverse" >>/etc/apt/sources.list
RUN echo "deb http://repo.huaweicloud.com/ubuntu/ focal-updates multiverse" >>/etc/apt/sources.list
RUN echo "deb http://repo.huaweicloud.com/ubuntu/ focal-backports main restricted universe multiverse" >>/etc/apt/sources.list
RUN echo "deb http://repo.huaweicloud.com/ubuntu focal-security main restricted" >>/etc/apt/sources.list
RUN echo "deb http://repo.huaweicloud.com/ubuntu focal-security multiverse" >>/etc/apt/sources.list
RUN apt-get update && apt-get upgrade -y
RUN ln -svf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime && apt-get install -y tzdata
RUN apt-get install --fix-missing -y build-essential curl cmake patchelf python3 python3-pip python3-dev ninja-build git wget gnupg2
RUN apt-get update

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 30
RUN update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 30
RUN pip config set global.index-url https://mirrors.cloud.tencent.com/pypi/simple/

RUN apt-get install -y pkg-config uuid-dev libpulse-dev

RUN python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install --upgrade numpy==1.20.3 && \
    pip3 install torch==1.11.0+cpu torchvision==0.12.0+cpu torchaudio==0.11.0+cpu \
     -f https://download.pytorch.org/whl/cpu/torch_stable.html && \
    python -m pip install --upgrade faiss-cpu && \
    python -m pip install --upgrade awscli awscli-plugin-endpoint && \
    echo OK: python
