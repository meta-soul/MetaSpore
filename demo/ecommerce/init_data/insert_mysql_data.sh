#!/bin/bash

INSERT_MYSQL_IAMAGE="metaspore-training-release-xhd-test:latest"
INSERT_MYSQL_CONTAINER_NAME="insert-mysql"
docker run -it --rm --network=host --volume "${PWD}/init_data/script:/opt/script" --name ${INSERT_MYSQL_CONTAINER_NAME} ${INSERT_MYSQL_IAMAGE} python /opt/script/insert_mysql_data.py
