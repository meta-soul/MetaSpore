一、先把依赖包安装好：

```
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/
```

二、运行 item 和 user 的 attr 计算以及灌库：

```
python pipeline_attr.py --conf conf/movielens_offline_attr.yaml
```

三、运行 item 和 user 的 emb 计算以及灌库：

由于目前 executor 没有安装依赖 python 第三方库，通过 local 模式进行测试：

```
spark-submit --master local --conf spark.jars.packages=org.mongodb.spark:mongo-spark-connector_2.12:3.0.1 pipeline_emb.py --conf conf/movielens_offline_emb.yaml
```

item embedding 计算目前支持 GPU/CPU 资源调用和 metaspore serving 两种模式，有关配置如下，建议几种模式都测一下：

```
device: "cuda"
infer_online: false  # false 则走 GPU/CPU 资源调用，具体是哪种计算资源由 device 控制；true 则走 serving 调用
infer_host: "172.31.0.197"
infer_port: 50001
infer_model: "all-MiniLM-L6-v2"
write_mode: overwrite
```
