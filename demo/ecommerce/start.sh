#!/bin/bash

# install metasporeflow
# pip install metasporeflow

# Create Mysql & MongoDB with docker compose
docker-compose -f init_container/create-database.yml up -d

echo "Waiting for mysql and mongodb to init db and user..."
sleep 10s

# insert data into mysql
./init_data/insert_mysql_data.sh

source ./env.sh
cd ecommerce_demo || exit
metaspore flow up
