## [中文介绍](README-CN.md)

# MetaSpore: One-stop machine learning development platform

MetaSpore is a one-stop end-to-end machine learning development platform that provides a full-cycle framework and development interface for from data preprocessing, model training, offline experiments, online predictions to online experiment traffic bucketization and ab-testing.

![MetaSpore Architecture](https://github.com/meta-soul/MetaSpore/raw/main/docs/images/MetaSpore-arch-en.jpg)

MetaSpore is developed and opensourced by [DMetaSoul](https://github.com/meta-soul?type=source) team. You could also join our [slack user discussion space](https://join.slack.com/t/dmetasoul-user/shared_invite/zt-1681xagg3-4YouyW0Y4wfhPnvji~OwFg).


## Core Features
MetaSpore has the following features:

1. One-stop end-to-end development, from offline model training to online prediction and bucketing experiments, with a unified development experience across the entire process;
2. Deep learning training framework, compatible with PyTorch ecology, supports distributed large-scale sparse feature learning;
2. The training framework is connected with PySpark to seamlessly read the training data from the data lake and data warehouse;
3. High-performance online prediction service, supports fast inference for neural network, decision tree, Spark ML, SKLearn and other models; supports heterogeneous hardware inference acceleration;
4. In the offline unified feature extraction framework, the online feature reading logic is automatically generated, and the feature extraction logic is unified cross offline and online;
5. Online algorithm application framework, providing model prediction, experiment bucketing and traffic splitting, dynamic hot loading of parameters and rich debug functions;
6. Rich industry algorithm examples and end-to-end solutions.

## Documentation and examples

* [Offline Training Getting Started Tutorial](https://github.com/meta-soul/MetaSpore/blob/main/tutorials/metaspore-getting-started.ipynb)

* [Online Algorithm Application (Java implementation)](https://github.com/meta-soul/MetaSpore/blob/main/java/online-serving/README.md)

    * [Online Model Serving](https://github.com/meta-soul/MetaSpore/blob/main/java/online-serving/serving/README.md)
    * [Online Feature Extraction](https://github.com/meta-soul/MetaSpore/blob/main/java/online-serving/feature-extract/README.md)
    * [Online Experiment Pipeline](https://github.com/meta-soul/MetaSpore/blob/main/java/online-serving/experiment-pipeline/README.md)

* [A MovieLens end-to-end recommender system demo](https://github.com/meta-soul/MetaSpore/blob/main/demo/movielens), including
    * [Offline models](https://github.com/meta-soul/MetaSpore/blob/main/demo/movielens/offline)
    * [Online algorithm application (Java implementation)](https://github.com/meta-soul/MetaSpore/blob/main/demo/movielens/online)

## Installation package download

### Training package
We provide precompiled offline training wheel package on pypi, install it via pip:
```bash
pip install metaspore
```
The minimum Python version required is 3.8.

After installation, also install pytorch and pyspark (they are not included as depenencies of metaspore wheel so you could choose pyspark and pytorch versions as needed):
```bash
pip install pyspark
pip install torch==1.11.0+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html
```

### Serving package
We provide prebuilt docker images for MetaSpore Serving Service:
#### CPU only image
```
docker pull dmetasoul/metaspore-serving-release:cpu-v1.0.1
```
#### GPU image
```
docker pull dmetasoul/metaspore-serving-release:gpu-v1.0.1
```

See [Run Serving Service in Docker](https://github.com/meta-soul/MetaSpore/blob/main/docs/run-serving-image.md) for details.

## Compile the code

* [Offline training framework compilation](https://github.com/meta-soul/MetaSpore/blob/main/docs/build-offline.md)

## Community guidelines
[Community guidelines](https://github.com/meta-soul/MetaSpore/blob/main/community-guideline.md)

## Feedback

For questions about usage, you can post questions in [GitHub Discussion](https://github.com/meta-soul/MetaSpore/discussions), or through [GitHub Issue](https://github.com/meta-soul/MetaSpore/issues).

### Mail
Email us at [opensource@dmetasoul.com](mailto:opensource@dmetasoul.com).

### Slack
Join our user discussion slack channel: [MetaSpore User Discussion](https://join.slack.com/t/dmetasoul-user/shared_invite/zt-1681xagg3-4YouyW0Y4wfhPnvji~OwFg)

## Open source projects
MetaSpore is a completely open source project released under the Apache License 2.0. Participation, feedback, and code contributions are welcome.