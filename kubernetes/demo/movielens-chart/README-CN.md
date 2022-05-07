# MOVIELENS CHART

Movielens 服务在 Kubernetes 上使用 helm 进行的一键部署

<br/>

## 使用方法

1. 定义环境变量
   ```
   export IMAGE_REPOSITORY=$YOUR-REPOSITORY/movielens
   export IMAGE_TAG=YOUR-TAG
   export NAMESPACE=YOUR-NAMESPACE
   # mongodb config
   export MONGODB_HOST=YOUR-MONGODB_HOST
   export MONGODB_PORT=27017
   export MONGODB_DATABASE=YOUR-MONGODB_DATABASE
   export MONGODB_USERNAME=YOUR-MONGODB_USERNAME
   export MONGODB_PASSWORD=YOUR-MONGODB_PASSWORD
   # milvus config
   export MILVUS_HOST=YOUR-MILVUS_HOST
   export MILVUS_PORT=19530
   
   ```
2. 执行安装命令进行 Movielens 服务的部署
   ```
   helm install movielens -n $NAMESPACE ./chart \
     --set image.repository=$IMAGE_REPOSITORY \
     --set image.tag=$IMAGE_TAG \
     --set mongodb.host=$MONGODB_HOST \
     --set mongodb.port=$MONGODB_PORT \
     --set mongodb.database=$MONGODB_DATABASE \
     --set mongodb.username=$MONGODB_USERNAME \
     --set mongodb.password=$MONGODB_PASSWORD \
     --set milvus.host=$MILVUS_HOST \
     --set milvus.port=$MILVUS_PORT
   
   ```