一、先把依赖包安装好：

```
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/
```

二、运行 item 和 user 的 attr 计算以及灌库：

```
python pipeline_attr.py --conf conf/movielens_offline_attr.yaml
```

三、运行 item 和 user 的 emb 计算以及灌库：

```
python pipeline_emb.py --conf conf/movielens_offline_emb.yaml
```

注：目前依赖 gpu，不能在 spark 上跑，待改为请求 serving 方式
