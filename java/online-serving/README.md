# Metaspore Online Serving

Metaspore Online Serving is an online application framework developed by [DMetaSoul](https://www.dmetasoul.com/) that supports online algorithm business services and low-code mode. Contains three sub-modules serving, experiment-pipeline, and feature-extract.

- serving: Model computing service, providing Sparse DNN model, twin tower model, Spark ML model, SKLearn model, XGBoost model prediction service and Milvus vector recall service.

- experiment-pipiline: Experiment Workflow SDK, a real-time dynamic update, low-code mode algorithm experiment flow framework. Supports common online functions such as dynamic update experiments, multi-layer orthogonality, and experimental cut flow.

- feature-extract: A framework for automatic feature generation, a framework for automatically generating JPA code for features based on Yaml files. Supports multiple DB sources and provides feature interfaces conveniently.

  

## Demo example

Complete example: [Movielens Demo example link](../../demo/movielens/online/README.md)



## Documentation

| Documentation | |
| :------------------- | :-----------------------------------------: |
| serving | [Doc](serving/README.md) |
| experiment-pipeline | [Doc](experiment-pipeline/README.md) |
| feature-extract | [Doc](feature-extract/README.md) |
| | |