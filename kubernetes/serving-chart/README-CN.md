# SERVING CHART

Serving 服务在 Kubernetes 上使用 helm 进行的一键部署

<br/>

## 使用方法

1. 定义环境变量
   ```
   export IMAGE_REPOSITORY=$YOUR-REPOSITORY/metaspore-serving-release
   export IMAGE_TAG=YOUR-TAG
   export NAMESPACE=YOUR-NAMESPACE
   export STORAGECLASS=YOUR-STORAGECLASS
   ```
2. 执行安装命令进行 Serving 服务的部署
   ```
   helm install serving -n $NAMESPACE ./chart \
     --set image.repository=$IMAGE_REPOSITORY \
     --set image.tag=$IMAGE_TAG \
     --set storageclass=$STORAGECLASS
   ```