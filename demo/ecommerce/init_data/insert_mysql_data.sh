#!/bin/bash

INSERT_MYSQL_IAMAGE="swr.cn-southwest-2.myhuaweicloud.com/dmetasoul-repo/metaspore-training-release:v1.1.0"
INSERT_MYSQL_CONTAINER_NAME="insert-mysql"
docker run -it --rm --network=host --volume "${PWD}/init_data/script:/opt/script" --name ${INSERT_MYSQL_CONTAINER_NAME} ${INSERT_MYSQL_IAMAGE} python /opt/script/insert_mysql_data.py
