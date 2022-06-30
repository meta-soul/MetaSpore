FROM python:3.10-slim-buster
RUN python -m pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
ADD . /metaspore-operator
RUN python -m pip install --no-cache-dir -r /metaspore-operator/requirements.txt
ENV INCLUSTER=1
CMD kopf run /metaspore-operator/metaspore_operator.py --verbose