#!/bin/bash
serving_bin="/opt/metaspore-serving/bin/metaspore-serving-bin"
recommend_cmd="java -Xmx2048M -Xms2048M -Xmn768M -XX:MaxMetaspaceSize=256M -XX:MetaspaceSize=256M -jar /opt/recommend-service.jar"
serving_grpc_port_name="-grpc_listen_port"
serving_grpc_port=50000
init_load_path_name="-init_load_path"
init_load_path="/data/models"
for i in "$@"
do
  next=$[ i + 1 ]
  echo "next:", $next
  if [ "$i" = "$serving_grpc_port_name" ]
  then
    if [ -n "$next" ]
    then
      serving_grpc_port=$next
    fi
  fi
  if [ "$i" = "$init_load_path_name" ]
  then
    if [ -n "$next" ]
    then
      init_load_path=$next
    fi
  fi
done

serving_cmd="${serving_bin} ${serving_grpc_port_name} ${serving_grpc_port} ${init_load_path_name} ${init_load_path} &"
echo "serving_cmd" ${serving_cmd}
echo "recommend_cmd" ${recommend_cmd}