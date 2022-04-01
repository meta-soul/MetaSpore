# Experiment Pipeline

Experiment Pipeline is a real-time dynamic update, low-code mode algorithm experiment flow framework SDK developed by [DMetaSoul](https://www.dmetasoul.com/). Users can describe and execute online algorithm experiment flow by configuring YAML files. call chain. It covers the algorithm system of multi-scene selection (Scenes), multi-level sorting (Layers), and multi-experiment ABTest (Experiments). Supports common online functions such as dynamic update experiments, multi-layer orthogonality, and experimental cut flow.



## Demo example

Full example: [Movielens Demo example link](../../../demo/movielens/online/README.md)



## Framework introduction

The framework is based on the spring boot project and dynamically updates the yaml file through consul.

- ### Yaml Introduction

  - Use Yaml to define the order and structure of Scenes, Layers, Experiments. Each Request comes in and you can select a scene (Scene) for recommendation. Multi-layer sorting (layer) is called according to the top-to-bottom order, e.g. recall -> sort. Experiments describe various properties of the experiment, such as parameters. Create config/test/scene-config on consul, and copy and paste the following Yaml into it:
  - Support parameter overlay: User-defined parameters are overridden according to key, and in the order of extraSceneArgs -> extraLayerArgs -> extraExperimentArgs; if the underlying parameters do not exist, the upper parameters are used.
  - Reflecting experiment examples against className is supported: if the experiment is configured with className (full classpath), it takes precedence to instantiate this class, and if not, it is instantiated according to the annotation name of the experiment.

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



- ### consul configuration

  In the spring boot project, the bootstrap.yml of consul needs to be configured and placed in the resources directory. Among them, prefix and defaultContext correspond to the directory on consul, and data-key is the key on consul. Copy the yaml file in the figure above to the value of consul. As shown below

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



- ### Import SDK

  Install and reference this project in your own spring boot project pom

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