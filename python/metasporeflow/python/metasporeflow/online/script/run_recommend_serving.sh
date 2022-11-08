#!/bin/bash
serving_bin="/opt/metaspore-serving/bin/metaspore-serving-bin"
recommend_base_cmd="java -Xmx2048M -Xms2048M -Xmn768M -XX:MaxMetaspaceSize=256M -XX:MetaspaceSize=256M -jar /opt/recommend-service.jar"
serving_grpc_port_name="-grpc_listen_port"
serving_grpc_port=50000
init_load_path_name="-init_load_path"
init_load_path="/data/models"
recommend_args=""
while [ $# -ne 0 ]
do
  if [ "$1" = "$serving_grpc_port_name" ] && [ -n "$2" ]
  then
    serving_grpc_port=$2
    shift 2
  elif [ "$1" = "$init_load_path_name" ] && [ -n "$2" ]
  then
    init_load_path=$2
    shift 2
  elif [ "$1" = "--CONSUL_ENABLE" ] && [ -n "$2" ]
  then
    recommend_args="${recommend_args} --CONSUL_ENABLE=$2"
    shift 2
  elif [ "$1" = "--AWS_ACCESS_KEY_ID" ] && [ -n "$2" ]
  then
    recommend_args="${recommend_args} --AWS_ACCESS_KEY_ID=$2"
    shift 2
  elif [ "$1" = "--AWS_SECRET_ACCESS_KEY" ] && [ -n "$2" ]
  then
    recommend_args="${recommend_args} --AWS_SECRET_ACCESS_KEY=$2"
    shift 2
  elif [ "$1" = "--AWS_ENDPOINT" ] && [ -n "$2" ]
  then
    recommend_args="${recommend_args} --AWS_ENDPOINT=$2"
    shift 2
  elif [ "$1" = "--SERVICE_PORT" ] && [ -n "$2" ]
  then
    recommend_args="${recommend_args} --SERVICE_PORT=$2"
    shift 2
  else
    shift
  fi
done
if [ -n "$init_load_path" ]
then
  mkdir -p $init_load_path
fi
serving_cmd="${serving_bin} ${serving_grpc_port_name} ${serving_grpc_port} ${init_load_path_name} ${init_load_path}"
echo "run serving_cmd" ${serving_cmd}
serving_log_file="/opt/metaspore_serving.log"
${serving_cmd} > $serving_log_file 2>&1 &
recommend_cmd="${recommend_base_cmd} ${recommend_args}"
echo "run recommend_cmd" ${recommend_cmd}
${recommend_cmd}
