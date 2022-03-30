# MovieLens Demo-推荐系统在线服务

在线的工程框架是建立在 SpringBoot+K8S 框架的基础上开发完成，它的扩展性很好，易于部署。我们可以按照一下的说明文档一步一步执行来配置好自己的开发环境。我们可以参考下面的系统架构图的说明，在这个Demo的系统里，我们在线的算法Pipeline，主要由用户建模、召回、排序、多样性、摘要获取等几个模块组成。当我们按照下述说明一步一步执行完之后，一个端到端的推荐系统将完整的呈现出来。

<p align="center">
   <img width="800" alt="image" src="https://user-images.githubusercontent.com/7464971/160770284-26bd3885-4d47-4c00-9260-b3dc1aeb4263.png">
</p>

## 1. 安装 online-serving 组件
首先， 我们需要使用 maven 来安装 [online-serving 组件](../../../java/online-serving/README-CN.md)：

1. `MetaSpore Serving` 用来做模型的实时推理，所有线上需要模型实时推理预测的工作都需要访问这个服务；

2. `feature-extract` 是一个 maven 的插件，用来通过[特征描述文件](src/main/resources/tables)，自动生成从 `MongoDB` 访问特征的API，更多信息可以[参考链接](../../../java/online-serving/feature-extract/README-CN.md)；

3. `experiment-pipeline` 是给 A/B 实验框架，能够帮助我们更方便的部署A/B实验，支持热部署等特性，更多信息可以[参考链接](../../../java/online-serving/experiment-pipeline/README-CN.md)。

安装时，需要执行以下命令
```shell
cd MetaSpore/java/online-serving
mvn clean install 
```

安装完上述组件之后，我们需要运行自动生成特征访问API的代码，执行如下命令：
```shell
cd MetaSpore/demo/movielens/online
mvn com.dmetasoul:feature-extract:1.0-SNAPSHOT:generate
```
最后，如果我们在 Intelli J IDEA 开发环境中，我们可以把目录 `target/generated-sources/feature/java` 标注成 `Generated Sources Root`。


## 2. 建立 application-dev.properties
我们需要从模版文件[application-template.properties](src/main/resources/application-template.properties) 创建一个 resources/**application-dev.properties** 文件，主要用来配置：
1. `MongoDB` 服务的相关配置；
2. `MetaSpore Serving` 服务的相关配置；
3. `Milvus` 服务的相关配置；
4. 其他服务，例如 `MySQL` 相关的配置可以忽略，我们在这个项目中暂时还未用到。

## 3. 安装和配置 Consul
我们可以通过更改 `Consul` 中的 Key/Value 的字典值，实时更改在线的 A/B 实验的策略，具体安装和配置的方法：
1. [下载并安装Consul](https://www.consul.io/downloads)，打开应用程序；
2. 打开Consul的[链接](http://localhost:8500/ui/dc1/kv), 默认端口是`8500`，如果遇到问题，可以查看是否存在端口占用的情况。创建一个新的 Key/Value 字典对:
   1. Key 是 `config/test/scene-config`
   2. 拷贝 [YAML 配置文件](src/main/resources/experiment.yaml) 这个文件中的内容作为 Value. 
   3. 在我们的Demo项目中，Consul 的[配置文件在这里](src/main/resources/bootstrap.yml).

## 4. 启动在线应用服务
当以上的配置工作都完成之后，我们可以从 `MovielensRecommendApplication.java` 这里作为服务的入口，启动我们的在线应用并进行测试。举例来说，对于 `userId=10` 的用户，我们可以通过
`curl http://localhost:8080/user/10` 这条命令来访问推荐服务，并获取推荐结果。

