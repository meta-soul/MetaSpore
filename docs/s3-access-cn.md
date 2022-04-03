# 配置 MetaSpore 离线训练 S3 访问权限

如果你也是使用云上对象存储来存储数据和模型，那么可以用 S3 协议来访问他们。除了 AWS S3，阿里云 OSS、华为云 OBS、腾讯云 COS 等都可以支持 S3 协议。因此 MetaSpore 只内置了 S3 的客户端来读写模型。

配置 S3 的访问权限有两种方式。

## 1. 在 AWS 服务器上访问 AWS S3
在 AWS 服务器上访问 S3，可以自动走 IAM Role 鉴权，不需要配置 access key 等环境变量参数。

备注：如果是中国区 AWS，那么需要增加 AWS_REGION=cn-north-1，否则 AWS SDK 默认是查找国际区的桶。可以在运行 MetaSpore 训练之前执行：
```bash
export AWS_REGION=cn-north-1
```

如果要运行分布式的 Spark 任务，则需要在代码中，创建 spark session 时，配置 Executor 环境变量：
```python
spark_session = SparkSession.builder
                .config('spark.executorEnv.AWS_REGION', 'cn-north-1')
                .getOrCreate()
```

## 2. 在非 AWS 云服务器上访问 S3/OSS/OBS

这种情况需要配置 AWS_ACCESS_KEY_ID、AWS_SECRET_ACCESS_KEY 鉴权环境变量。对于不是 AWS S3 的云存储（OSS、OBS、COS 等），还需要配置 AWS_ENDPOINT。示例：
```bash
export AWS_ENDPOINT=<end point url>
export AWS_ACCESS_KEY_ID=<your access key id>
export AWS_SECRET_ACCESS_KEY=<your access key>
```

运行分布式任务配置方法同上。

具体的 endpoint url，可以登录你的云存储控制台查看，也可以参考对应云商的文档：

https://docs.amazonaws.cn/aws/latest/userguide/endpoints-Beijing.html

https://help.aliyun.com/document_detail/31837.html

https://cloud.tencent.com/document/product/436/6224

https://support.huaweicloud.com/productdesc-obs/obs_03_0152.html