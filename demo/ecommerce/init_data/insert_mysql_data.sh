#!/bin/bash

INSERT_MYSQL_IAMAGE=$OFFLINE_IMAGE
INSERT_MYSQL_CONTAINER_NAME="metaspore-insert-mysql"
docker run -it --rm --network=host --volume "${PWD}/init_data/script:/opt/script" --name ${INSERT_MYSQL_CONTAINER_NAME} ${INSERT_MYSQL_IAMAGE} python /opt/script/insert_mysql_data.py
