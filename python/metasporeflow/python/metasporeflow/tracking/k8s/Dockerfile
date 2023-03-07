FROM python:3.8.16

RUN /usr/local/bin/python -m pip install -i https://repo.huaweicloud.com/repository/pypi/simple/ --upgrade pip 
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple/ awscli watchdog aiohttp
RUN pip install -i https://repo.huaweicloud.com/repository/pypi/simple/ esdk-obs-python --trusted-host pypi.org
COPY ./src /opt/src
RUN chmod +x /opt/src/entrypoint.py
WORKDIR /opt/src
CMD ["/usr/local/bin/python", "/opt/src/entrypoint.py"]
