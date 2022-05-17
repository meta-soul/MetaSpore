#!/bin/bash

set -x
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
PROTO_DIR=$DIR/../../../protos
# generate proto files
python -m grpc_tools.protoc -I $PROTO_DIR --python_out=$DIR --grpc_python_out $DIR $PROTO_DIR/metaspore.proto

docker build -t consul-watch-load:v1.0.0 -f $DIR/Dockerfile $(dirname "$0")