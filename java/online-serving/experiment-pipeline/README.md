## 使用教程
1. 创建 consul 目录：config/test/scene-config
2. 复制下面配置到 consul 里 scene-config 下面的 value
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
3. 在自己的 spring boot 项目里 resources 目录下配置 consul 的配置文件
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
4. 安装并在自己的 spring boot 项目 pom 中引用本项目
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