#!/bin/bash

source ./env.sh

docker compose -f init_container/create-database.yml down

cd ecommerce_demo || exit
metaspore flow down
