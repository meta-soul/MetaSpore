# 1. 简介
MetaSpore Serving 服务是一个 C++ 实现的模型推理服务，支持 MetaSpore 训练生成的 Sparse DNN 模型，同时也支持 XGBoost、LightGBM、SKLearn、HuggingFace 等各类模型的在线推理计算。Serving 服务提供了 gRPC 远程调用的接口，可以使用 Java、Python 等语言作为客户端来调用模型推理计算。

# 2. 保存模型到本地
以 xgboost 为例，训练模型并保存为 onnx 格式：

```python
import xgboost as xgb
import numpy as np
import pathlib
import os

data = np.random.rand(5, 10).astype('f')  # 5 entities, each contains 10 features
label = np.random.randint(2, size=5)  # binary target
dtrain = xgb.DMatrix(data, label=label)

param = {'max_depth': 2, 'eta': 1, 'objective': 'binary:logistic'}
param['nthread'] = 4
param['eval_metric'] = 'auc'

num_round = 10
bst = xgb.train(param, dtrain, num_round, )

from onnxmltools import convert_xgboost
from onnxconverter_common.data_types import FloatTensorType

initial_types = [('input', FloatTensorType(shape=[-1, 10]))]
xgboost_onnx_model = convert_xgboost(bst, initial_types=initial_types, target_opset=14)

output_dir = "output/model_export/xgboost_model/"

pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
pathlib.Path(os.path.join(output_dir, 'dense')).mkdir(parents=True, exist_ok=True)

with open(os.path.join(output_dir, 'dense/model.onnx'), "wb") as f:
    f.write(xgboost_onnx_model.SerializeToString())
with open(os.path.join(output_dir, 'dense_schema.txt'), "w") as f:
    f.write('table: input\n')
```

# 3. 通过 docker 镜像启动 MetaSpore Serving 服务
启动 docker 容器，并将宿主机上的模型目录 ${PWD}/output/model_export 映射到容器 /data/models，注意宿主机目录是上一步生成的 xgboost_model 模型目录的上级目录。
```bash
docker run -d --name=test-serving --net host -v ${PWD}/output/model_export:/data/models swr.cn-southwest-2.myhuaweicloud.com/dmetasoul-public/metaspore-serving-release:cpu-v1.0.1 /opt/metaspore-serving/bin/metaspore-serving-bin -grpc_listen_port 50000 -init_load_path /data/models
```
  * **注** ：如果需要使用 GPU 进行预测，则 docker 镜像地址为：
    ``` 
    swr.cn-southwest-2.myhuaweicloud.com/dmetasoul-public/metaspore-serving-release:gpu-v1.0.1
    ```
    启动服务时，需要在 `docker run` 后面需要加上 `--gpus all` 参数。宿主机上需要提前安装好 nvidia-docker-plugin。

启动后可以查看模型是否加载成功：
```bash
docker logs test-serving
```
日志中包含输出
TabularModel loaded from /data/models/xgboost, required inputs [input], producing outputs [label, probabilities]
说明 xgboost 模型加载成功。
日志中包含 Use cuda:0 说明服务识别到了 GPU 并自动用 GPU 进行预测。

# 4. 调用 Serving 服务

## 4.1 Python
安装依赖：
```bash
pip install grpcio-tools pyarrow
```
生成 grpc python 定义文件：
```bash
# MetaSpore 需要替换为 MetaSpore 代码目录
python -m grpc_tools.protoc -I MetaSpore/protos/ --python_out=. --grpc_python_out . MetaSpore/protos/metaspore.proto
```

调用 Serving 服务示例 Python 代码：
```python
import grpc

import metaspore_pb2
import metaspore_pb2_grpc

import pyarrow as pa

with grpc.insecure_channel('0.0.0.0:50000') as channel:
    stub = metaspore_pb2_grpc.PredictStub(channel)
    row = []
    values = [0.6558618,0.13005558,0.03510657,0.23048967,0.63329154,0.43201634,0.5795548,0.5384891,0.9612295,0.39274803]
    for i in range(10):
        row.append(pa.array([values[i]], type=pa.float32()))
    rb = pa.RecordBatch.from_arrays(row, [f'field_{i}' for i in range(10)])
    sink = pa.BufferOutputStream()
    with pa.ipc.new_file(sink, rb.schema) as writer:
        writer.write_batch(rb)
    payload_map = {"input": sink.getvalue().to_pybytes()}
    request = metaspore_pb2.PredictRequest(model_name="xgboost_model", payload=payload_map)
    reply = stub.Predict(request)
    for name in reply.payload:
        with pa.BufferReader(reply.payload[name]) as reader:
            tensor = pa.ipc.read_tensor(reader)
            print(f'Tensor: {tensor.to_numpy()}')
```

## 4.2 Java

可以在本地执行 https://github.com/meta-soul/MetaSpore/blob/main/java/online-serving/serving/src/test/java/com/dmetasoul/metaspore/serving/DenseXGBoostTest.java 这个测试来调用 xgboost 模型预测。


## 5. 生产部署 Serving 服务

我们提供了 K8s Helm Chart ：
https://github.com/meta-soul/MetaSpore/tree/main/kubernetes/serving-chart