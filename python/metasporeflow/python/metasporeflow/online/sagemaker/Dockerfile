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
ARG SERVING_IMAGE
ARG RECOMMEND_IMAGE
ARG RUNTIME=cpu
ARG SERVICE_PORT=8080
FROM nvidia/cuda:11.6.2-cudnn8-runtime-ubuntu20.04 as serving_gpu_base
FROM ubuntu:20.04 as serving_cpu_base

FROM serving_${RUNTIME}_base as serving_base
ENV DEBIAN_FRONTEND=noninteractive
RUN echo "deb http://repo.huaweicloud.com/ubuntu/ focal main restricted" >/etc/apt/sources.list
RUN echo "deb http://repo.huaweicloud.com/ubuntu/ focal-updates main restricted" >>/etc/apt/sources.list
RUN echo "deb http://repo.huaweicloud.com/ubuntu/ focal universe" >>/etc/apt/sources.list
RUN echo "deb http://repo.huaweicloud.com/ubuntu/ focal-updates universe" >>/etc/apt/sources.list
RUN echo "deb http://repo.huaweicloud.com/ubuntu/ focal multiverse" >>/etc/apt/sources.list
RUN echo "deb http://repo.huaweicloud.com/ubuntu/ focal-updates multiverse" >>/etc/apt/sources.list
RUN echo "deb http://repo.huaweicloud.com/ubuntu/ focal-backports main restricted universe multiverse" >>/etc/apt/sources.list
RUN echo "deb http://repo.huaweicloud.com/ubuntu focal-security main restricted" >>/etc/apt/sources.list
RUN echo "deb http://repo.huaweicloud.com/ubuntu focal-security universe" >>/etc/apt/sources.list
RUN echo "deb http://repo.huaweicloud.com/ubuntu focal-security multiverse" >>/etc/apt/sources.list
RUN apt-get update && apt-get upgrade -y && apt-get clean
RUN apt-get update && apt-get install -y zip unzip curl wget ca-certificates && apt-get clean
RUN apt-get install -y locales && apt-get clean
RUN sed -i '/en_US.UTF-8/s/^# //g' /etc/locale.gen && \
    locale-gen
ENV LANG=en_US.UTF-8
ENV LANGUAGE=en_US.UTF-8
ENV LC_CTYPE=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8
ENV TZ=Asia/Shanghai
RUN ln -svf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime && apt-get install -y tzdata && apt-get clean

RUN apt-get install --fix-missing -y python3 python3-venv python3-pip python-is-python3 ca-certificates openjdk-11-jre-headless && apt-get clean

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 30
RUN python -m pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple/
RUN python -m pip config set globa.extra-index-url "https://pypi.tuna.tsinghua.edu.cn/simple/ https://pypi.org/simple/"

FROM ${SERVING_IMAGE} as serving_stage
FROM ${RECOMMEND_IMAGE} as recommend_stage

FROM serving_base as sagemaker_release
COPY --from=serving_stage /opt/metaspore-serving/ /opt/metaspore-serving
COPY --from=recommend_stage recommend-service-*-SNAPSHOT.jar /opt/recommend-service.jar

COPY dockerd-entrypoint.py /usr/local/bin/dockerd-entrypoint.py
RUN chmod +x /usr/local/bin/dockerd-entrypoint.py

ENV OMP_NUM_THREADS=1

ENTRYPOINT ["python", "/usr/local/bin/dockerd-entrypoint.py"]

CMD ["serve"]
