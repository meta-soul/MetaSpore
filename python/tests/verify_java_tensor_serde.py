import metaspore_pb2
import pyarrow as pa

with open('arrow_tensor_java_ser.bin', 'rb') as f:
    message = metaspore_pb2.PredictRequest()
    message.ParseFromString(f.read())
    for name in message.payload:
        print(f'{name}')
        with pa.BufferReader(message.payload[name]) as reader:
            tensor = pa.ipc.read_tensor(reader).to_numpy()
            print(f'Tensor: {tensor}, shape: {tensor.shape}')