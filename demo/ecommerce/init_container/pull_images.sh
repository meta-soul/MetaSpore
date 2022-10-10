#!/bin/bash
docker_iamge_list=($MONGODB_IMAGE $MYSQL_IMAGE $OFFLINE_IMAGE $ONLINE_IMAGE $FRONT_IMAGE)

for image in ${docker_iamge_list[@]}; do
    sudo docker pull $image
done