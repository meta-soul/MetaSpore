#!/bin/bash
serving_bin="/opt/metaspore-serving/bin/metaspore-serving-bin"
recommend_base_cmd="java -Xmx2048M -Xms2048M -Xmn768M -XX:MaxMetaspaceSize=256M -XX:MetaspaceSize=256M -jar /opt/recommend-service.jar"
serving_grpc_port_name="-grpc_listen_port"
serving_grpc_port=50005
init_load_path_name="-init_load_path"
init_load_path="/data/models"
aws_access_key_id=$AWS_ACCESS_KEY_ID
aws_secret_access_key=$AWS_SECRET_ACCESS_KEY
aws_endpoint=$AWS_ENDPOINT
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
    aws_access_key_id=$2
    shift 2
  elif [ "$1" = "--AWS_SECRET_ACCESS_KEY" ] && [ -n "$2" ]
  then
    aws_secret_access_key=$2
    shift 2
  elif [ "$1" = "--AWS_ENDPOINT" ] && [ -n "$2" ]
  then
    aws_endpoint=$2
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
mkdir -p /root/.aws
aws_config="/root/.aws/config"
(
cat << EOF
[default]
s3 =
    addressing_style = virtual
    max_bandwidth = 50MB/s
    endpoint_url = http://${aws_endpoint}
[plugins]
endpoint = awscli_plugin_endpoint
EOF
) > $aws_config

aws_credentials="/.aws/credentials"
(
cat << EOF
[default]
aws_access_key_id = $aws_access_key_id
aws_secret_access_key = $aws_secret_access_key
EOF
) > $aws_credentials
serving_cmd="${serving_bin} ${serving_grpc_port_name} ${serving_grpc_port} ${init_load_path_name} ${init_load_path}"
echo "run serving_cmd" ${serving_cmd}
serving_log_file="/opt/metaspore_serving.log"
${serving_cmd} > $serving_log_file 2>&1 &
echo "please wait model serving start!"
sleep 10s
recommend_cmd="${recommend_base_cmd} ${recommend_args}"
echo "run recommend_cmd" ${recommend_cmd}
${recommend_cmd}
