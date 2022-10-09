# metasporeflow-ecommerce-demo

## Getting Started
### Dependencies
- Docker
- Docker Compose
- Python 3.8+

1. Install MetaSpore Flow Cli
```bash
python -m pip install metasporecli==0.1.0
```

2. Run the ecommerce demo using metasporeflow including offline-flow/online-flow
```bash
./start.sh
```

3. When the program is completed, view the recommended results on the browser. 
```bash
# [Option] If you run this demo on a remote development machine, you can establish an ssh tunnel
ssh -L 41730:localhost:41730 -i <your private key> <your username>@<your remote ip>

# Then, you can access the web UI on your local machine
http://localhost:41730
```

4. Stop the ecommerce demo including releasing resources and docker containers
```bash
./stop.sh
```

## Demo Instructions 

### Init Container

Create Mysql & MongoDB with docker compose
```bash
docker compose -f init_container/create-database.yml up -d
```

Check Mysql & MongoDB status
```bash
docker ps | grep metaspore-local
```

Stop Mysql & MongoDB container
```bash
docker compose -f init_container/create-database.yml down
```

### Insert MYSQL

```bash
./init_data/insert_mysql_data.sh
```

## Metaspore Flow Cli

- start metasporeflow
```bash
source ./env.sh
cd ecommerce_demo
metaspore flow up
```

- stop metasporeflow
```bash
source ./env.sh
cd ecommerce_demo
metaspore flow down
```

## Offline Training DAG
```
start
-> sync_data
-> join_data 
-> train_model_pop & train_model_itemcf & train_model_deepctr 
-> notify_load_model
-> end
```
|     Node of DAG     | Description                                                                  |
|:-------------------:|:-----------------------------------------------------------------------------|
|      sync_data      | Convert User, Item and Interaction tables from MySQL to local parquet files. |
|      join_data      | Join three dataframes together and generate features.                        |
|   train_model_pop   | Prepare the most popular items for retrieval.                                |
| train_model_itemcf  | Train item based collaborative filtering for retrieval.                      |
| train_model_deepctr | Train CTR model - Wide&Deep for ranking.                                     |
|  notify_load_model  | Notify online serving that all offline models are ready.                     |
