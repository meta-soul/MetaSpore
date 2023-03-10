FROM python:3.8.16

RUN /usr/local/bin/python -m pip install -i https://repo.huaweicloud.com/repository/pypi/simple/ --upgrade pip 
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple/ awscli watchdog aiohttp
RUN pip install -i https://repo.huaweicloud.com/repository/pypi/simple/ esdk-obs-python --trusted-host pypi.org
RUN mkdir -p /tmp/tracking
RUN chmod 777 /tmp/tracking
COPY . /opt/src
RUN chmod +x /opt/src/entrypoint.py
WORKDIR /opt/src
CMD ["/bin/sh", "-c", "python /opt/src/entrypoint.py > /opt/entrypoint.log 2>&1"]
