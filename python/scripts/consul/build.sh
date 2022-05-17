#!/bin/bash
set -x
set -e

# generate proto files
python -m grpc_tools.protoc -I ../../../protos/ --python_out=. --grpc_python_out . ../../../protos/metaspore.proto

docker build -t consul-watch-load:v1.0.0 .