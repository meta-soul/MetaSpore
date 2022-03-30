# Experiment Pipeline

Experiment Pipeline is a real-time dynamic update, low-code mode algorithm experiment flow framework SDK developed by [DMetaSoul](https://www.dmetasoul.com/). Users can describe and execute online algorithm experiment flow by configuring YAML files. call chain. It covers the algorithm system of multi-scene selection (Scenes), multi-level sorting (Layers), and multi-experiment ABTest (Experiments). Supports common online functions such as dynamic update experiments, multi-layer orthogonality, and experimental cut flow.



## Demo example

Full example: [Movielens Demo example link](../../../demo/movielens/online/README.md)



## Framework introduction

The framework is based on the spring boot project and dynamically updates the yaml file through consul.

- ### Yaml Introduction

  Use Yaml to define the order and structure of Scenes, Layers, Experiments. Each Request comes in and you can select a scene (Scene) for recommendation. Multi-layer sorting (layer) is called according to the top-to-bottom order, e.g. recall -> sort. Experiments describe various properties of the experiment, such as parameters. Create config/test/scene-config on consul, and copy and paste the following Yaml into it:

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
        <groupId>com.dmetasoul</groupId>
        <artifactId>experiment-pipeline</artifactId>
        <version>1.0-SNAPSHOT</version>
    </dependency>
    ```