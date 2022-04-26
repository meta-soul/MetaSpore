# Sentence Embedding Training Pipeline

**Sentence Embedding** 模型是实现语义检索的模型基石，决定了检索效果的优劣，这里提供了完整的一套 Sentence Embedding 模型离线训练调优的代码，含训练（有监督/无监督）、蒸馏、评测、导出。

整个离线模型调优过程都是可配置低代码模式的，只要指定一些参数就可以完成离线优化流程的装配，这里以 [LCQMC 问题匹配数据集](https://www.luge.ai/#/luge/dataDetail?id=14)为例来介绍如何完成模型离线训练优化：

1. 创建离线实验配置文件，这里已经创建好了 `conf/lcqmc_experiment.yaml`，可以直接使用，如果有自定义需求可以参照该文件进行创建

2. 生成离线优化的 pipeline 配置文件

    ```
    python src/pipeline/make.py --exp-yaml conf/lcqmc_experiment.yaml --pipe-yaml conf/lcqmc_pipeline.yaml
    ```

    生成的 `conf/lcqmc_pipeline.yaml` 配置文件中定义了，整个 pipeline 流程包含几个环节，后续离线优化的启动将参照该文件执行。

3. 启动离线优化 pipeline

    ```
    nohup python src/pipeline/run.py conf/lcqmc_pipeline.yaml > lcqmc_pipeline.log 2>&1 &
    ```

    可以通过 `lcqmc_pipeline.log` 来查看当前优化进度，最终各阶段训练结果将保存在 `output/lcqmc/v1` 目录
