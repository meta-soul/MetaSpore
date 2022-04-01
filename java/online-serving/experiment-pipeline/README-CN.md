# Experiment Pipeline

Experiment Pipeline 是由 [DMetaSoul](https://www.dmetasoul.com/) 研发的实时动态更新、低代码模式算法实验流框架 SDK，用户可以通过配置 YAML 文件，来描述和执行线上算法实验流程的调用链。涵盖多场景选择（Scenes）、多层级排序（Layers）、多实验ABTest（Experiments）的算法体系。支持动态更新实验、多层正交、实验切流等线上常用功能。



## Demo 示例

完整示例：[Movielens Demo 示例链接](../../../demo/movielens/online/README.md)



## 框架介绍

框架基于 spring boot 项目，通过 consul 动态更新 yaml 文件。

### Yaml 介绍

- 使用 Yaml 来定义 Scenes、Layers、Experiments 的顺序和结构。每个 Request 进来可以选择某个场景（Scene）进行推荐。多层排序（layer）根据从上至下的顺序进行调用，例如 召回 -> 排序。实验（experiments）则描述了实验的各种属性，比如参数等。 在 consul 上创建 config/test/scene-config ，并把下图 Yaml 复制粘贴进去：
- 支持参数逐级覆盖：用户自定义参数会根据 key，并按照 extraSceneArgs -> extraLayerArgs -> extraExperimentArgs 的顺序以此覆盖；如果底层参数不存在，则使用上层参数。
- 支持根据 className 反射实验示例：如果实验配置了 className （完整类路径），则优先实例化此类；如果没有配置，则会根据实验的注解名进行实例化。

```yaml
scene-config:
  scenes:
    - name: guess-you-like
      extraSceneArgs:
        extra-arg1: sceneArg1
        extra-arg2: sceneArg2
      layers:
        - name: recall
          normalLayerArgs:
            - experimentName: RecallExperimentOne
              ratio: 1.0
            - experimentName: RecallExperimentTwo
              ratio: 0
          extraLayerArgs:
            extra-arg1: recallLayerArg1
        - name: rank
          normalLayerArgs:
            - experimentName: RankExperimentOne
              ratio: 1.0
            - experimentName: RankExperimentTwo
              ratio: 1.0
          extraLayerArgs:
            extra-arg1: rankLayerArg1
  experiments:
    - layerName: recall
      experimentName: RecallExperimentOne
      extraExperimentArgs:
        modelName: RecallExperimentOneModel
        extra-arg1: RecallExperimentOne-extra-arg1
    - layerName: recall
      experimentName: RecallExperimentTwo
      extraExperimentArgs:
        modelName: RecallExperimentTwoModel
        extra-arg1: RecallExperimentTwo-extra-arg1
    - layerName: rank
      experimentName: RankExperimentOne
      extraExperimentArgs:
        modelName: RankExperimentOneModel
        extra-arg1: RankExperimentOne-extra-arg1
    - layerName: rank
      experimentName: RankExperimentTwo
      className: com.dmetasoul.metaspore.example.experiment.RankExperimentOne
      extraExperimentArgs:
        modelName: RankExperimentOneModel
        extra-arg1: RankExperimentTwo-extra-arg1
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
    <groupId>com.dmetasoul.metaspore</groupId>
    <artifactId>experiment-pipeline</artifactId>
    <version>1.0-SNAPSHOT</version>
</dependency>
```