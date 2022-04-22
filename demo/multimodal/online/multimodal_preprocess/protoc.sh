python -m grpc_tools.protoc -I./proto --python_out=./hf_preprocessor --grpc_python_out=./hf_preprocessor proto/hf_preprocessor.proto

# You should modify the import statements in hf_preprocessor/hf_preprocessor_pb2_grpc.py

