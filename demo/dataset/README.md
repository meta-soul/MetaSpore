# Data Processing and Preparation
In this project, data processing is unified for movieLens-1m, movielens-25m, criteo-5d and other datasets, including feature generation,  match dataset generation, rank dataset generation, negative sampling, etc.

## Initialize the Configuration Files for Models
Before we continue to dive into the offline models, we should firstly initialize the config files from their YAML template for substituting some variables. For example
```shell
export MY_S3_BUCKET='your S3 bucket directory'
envsubst < template.yaml > output.yaml 
```
For the latter stages, we assume that we have done this to generate the available configurations before running the python scripts. If you have not install `envsubst`, you can run `sudo apt-get install gettext-base` assumming you are running this scripts on Debian based Linux systems.

## MovieLens-1M
In this section, we use [MoiveLens-1M](https://grouplens.org/datasets/movielens/1m/) to demonstrate. You can download these datasets from the provided urls and store these files onto your own cloud strorage. 

### 1. Feature Generation

Assuming we are in root directory of this project, we can execute the following commands to get the result of feature generation processing.

 ```shell
 cd ml_1m
 python fg.py --conf fg.yaml --verbose
 ```

### 2. Match Dataset
Assuming we are in root directory of this project, we can execute the following commands to get the train and test dataset of `Collaborative Filtering` algorithms.

```shell
cd ml_1m
python match_dataset_cf.py --conf match_dataset.yaml --verbose
```

After that we can get the train and test dataset of `TwoTowers` models through following command:

```shell
python match_dataset_negsample.py --conf match_dataset_negsample_10.yaml --verbose
```

### 3. Rank Dataset
Assuming we are in root directory of this project, we can execute the following commands to get the train and test dataset of `CTR estimator` models:

```shell
python rank_dataset.py --conf rank.yaml --verbose
```

## MovieLens-25M
In this section, we use [MoiveLens-25M](https://grouplens.org/datasets/movielens/25m/) to demonstrate. You can download these datasets from the provided urls and store these files onto your own cloud strorage.

### 1. Feature Generation

Assuming we are in root directory of this project, we can execute the following commands to get the result of feature generation processing.

 ```shell
 cd ml_25m
 python fg.py --conf fg.yaml --verbose
 ```

### 2. Match Dataset
Assuming we are in root directory of this project, we can execute the following commands to get the train and test dataset of `TwoTowers` models.

```shell
cd ml_25m
python match_dataset_negsample.py --conf match_dataset_negsample_10.yaml --verbose
```

### 3. Rank Dataset
Assuming we are in root directory of this project, we can execute the following commands to get the train and test dataset of `CTR estimator` models:

```shell
python rank_dataset.py --conf rank.yaml --verbose
```
