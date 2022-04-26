## [中文介绍](README-CN.md)

# Sentence Embedding Training Pipeline

The **Sentence Embedding** model is the basestone for semantic retrieval, which determines the quality of the retrieval system. Here is a whole of model offline training pipeline, including training (supervised/unsupervised), distillation, evaluation, export, etc.

The offline pipeline is configurable in low-code mode. The assembly of the offline pipeline can be completed as long as some configurable parameters are specified. Here, the [LCQMC problem matching data set](https://www.luge.ai/#/luge/dataDetail?id=14) as an example to introduce how to complete the offline training and optimization of the model:

1. Create an offline experiment configuration file. Because `conf/lcqmc_experiment.yaml` has been created here, which can be used directly. If you have custom requirements, you can refer to this file to create own.

2. Make a pipeline yaml based the above experiment config file

    ```
    python src/pipeline/make.py --exp-yaml conf/lcqmc_experiment.yaml --pipe-yaml conf/lcqmc_pipeline.yaml
    ```

    the generated `conf/lcqmc_pipeline.yaml` file includes all of stages of your offline pipeline

3. Start the offline pipeline via config file

    ```
    nohup python src/pipeline/run.py conf/lcqmc_pipeline.yaml > lcqmc_pipeline.log 2>&1 &
    ```

    you can tail the log file `lcqmc_pipeline.log` to check the progress of pipeline
