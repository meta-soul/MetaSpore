import traceback
from dataclasses import dataclass
import os

@dataclass(frozen=True)
class ModelInfo:
    name: str
    service: str
    path: str
    version: str
    util_cmd: str

def notify_loading_model(model_info):
    import grpc
    import metaspore_pb2
    import metaspore_pb2_grpc
    print('Notify loading model %s' % model_info)
    local_path = "/data/models/ctr/nn/widedeep/model_export/" + os.path.basename(model_info.path)
    try:
        with grpc.insecure_channel('127.0.0.1:50000') as channel:
            stub = metaspore_pb2_grpc.LoadStub(channel)
            request = metaspore_pb2.LoadRequest(model_name=model_info.name, version=model_info.version, dir_path=local_path)
            reply = stub.Load(request)
            print('OK: %s' % reply.msg)
            return True
    except Exception:
        traceback.print_exc()
        print('Fail to notify loading model %s, local_path: %s' % (model_info, local_path))
        raise



if __name__ == '__main__':
    import argparse
    import yaml
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, required=True)
    args = parser.parse_args()
    
    spec = dict()
    with open(args.conf, 'r') as stream:
        yaml_dict = yaml.load(stream, Loader=yaml.FullLoader)
        spec = yaml_dict['spec']

    notify_path = spec['training']['estimator_params']['model_export_path']
    experiment_name = spec['training']['estimator_params']['experiment_name']
    model_version = spec['training']['estimator_params']['model_version']
    
    model_info = ModelInfo(name=experiment_name,
                       service=experiment_name,
                       path=notify_path,
                       version=model_version,
                       util_cmd='echo OK')
    notify_loading_model(model_info)
