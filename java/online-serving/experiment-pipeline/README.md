## 使用教程
1. 创建 consul 目录：config/test/scene-config
2. 复制下面配置到 consul 里 scene-config 下面的 value
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