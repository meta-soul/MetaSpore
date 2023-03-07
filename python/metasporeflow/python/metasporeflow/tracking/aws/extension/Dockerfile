FROM python:3.8-alpine AS installer
#Layer Code
COPY extensionssrc /opt/
COPY extensionssrc/requirements.txt /opt/
RUN pip install -r /opt/requirements.txt -t /opt/extensions/lib -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade
WORKDIR /opt/extensions
RUN chmod -R 755 *.py

FROM scratch AS base
WORKDIR /opt/extensions
COPY --from=installer /opt/extensions .
