FROM python:3.10
RUN python -m pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
ADD . /metaflow-operator
RUN python -m pip install --no-cache-dir -r /metaflow-operator/requirements.txt
ENV INCLUSTER=1
CMD kopf run /metaflow-operator/metaflow_operator.py --verbose