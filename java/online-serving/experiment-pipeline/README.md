# Experiment Pipeline

Experiment Pipeline 是由 [DMetaSoul](https://www.dmetasoul.com/) 研发的实时动态更新、低代码模式算法实验流框架 SDK，用户可以通过配置 YAML 文件，来描述和执行线上算法实验流程的调用链。涵盖多场景选择（Scenes）、多层级排序（Layers）、多实验ABTest（Experiments）的算法体系。支持动态更新实验、多层正交、实验切流等线上常用功能。



## Demo 示例

完整示例：[Movielens Demo 示例链接](https://github.com/meta-soul/MetaSpore/blob/main/demo/movielens/online/README.md)



## 框架介绍

框架基于 spring boot 项目，通过 consul 动态更新 yaml 文件。

- ### Yaml 介绍

  使用 Yaml 来定义 Scenes、Layers、Experiments 的顺序和结构。每个 Request 进来可以选择某个场景（Scene）进行推荐。多层排序（layer）根据从上至下的顺序进行调用，例如 召回 -> 排序。实验（experiments）则描述了实验的各种属性，比如参数等。 在 consul 上创建 config/test/scene-config ，并把下图 Yaml 复制粘贴进去：

```yaml
scene-config:
  scenes:
    - name: guess-you-like
      sceneArgs:
        sceneArgs1: sceneArgs1-value
        sceneArgs2: sceneArgs2-value
      layers:
        - name: recall
          normalLayerArgs:
            - experimentName: milvus
              ratio: 1.0
            - experimentName: milvus2
              ratio: 0
          extraLayerArgs:
            extraLayerArgs1: extraLayerArgs1-value
            extraLayerArgs2: extraLayerArgs2-value
        - name: rank
          normalLayerArgs:
            - experimentName: milvus3
              ratio: 1.0
          extraLayerArgs:
            extraLayerArgs1: extraLayerArgs1-value
            extraLayerArgs2: extraLayerArgs2-value
            extraLayerArgs3: extraLayerArgs3-value
  experiments:
    - layerName: recall
      experimentName: milvus
      experimentArgs:
        modelName: TwoTower
        extraArg1: milvus-value
    - layerName: recall
      experimentName: milvus2
      experimentArgs:
        modelName: TwoTower2
        extraArg1: milvus2-value
        extraArg2: milvus2-value
    - layerName: recall
      experimentName: milvus4
      experimentArgs:
        modelName: TwoTower2
        extraArg1: milvus4-value
        extraArg2: milvus4-value
        extraArg3: milvus4-value
        extraArg4: milvus4-value
    - layerName: rank
      experimentName: milvus3
      experimentArgs:
        modelName: TwoTower3
        extraArg1: milvus3-value
        extraArg2: milvus3-value
        extraArg3: milvus4-value
```



- ### consul 配置

在 spring boot 项目里需要配置 consul 的  bootstrap.yml，放在 resources 目录下。其中 prefix、defaultContext 对应的是 consul 上的目录，data-key 为 consul 上的 key，把上图的 yaml 文件 复制到 consul 的value 里面即可。如下图

```yaml
spring:
  cloud:
    consul:
      host: localhost
      port: 8500
      config:
        enabled: true
        prefix: config
        defaultContext: test
        data-key: scene-config
        format: YAML
        watch:
          wait-time: 1
          delay: 1000
        fail-fast: true
```



- ### 导入 SDK

安装并在自己的 spring boot 项目 pom 中引用本项目

```shell
cd experiment-pipeline
mvn clean install
```
```xml
<dependency>
    <groupId>com.dmetasoul</groupId>
    <artifactId>experiment-pipeline</artifactId>
    <version>1.0-SNAPSHOT</version>
</dependency>
```