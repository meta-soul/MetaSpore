from modelarts.session import Session
from modelarts.model import Model
from modelarts.config.model_config import ServiceConfig

from modelarts.session import Session

access_key = 'NWXCHCV3JL3JSSXIMBEN'
secret_key = 'qywejD1gE1J4fvd2japTwq8QNvt5aZhcWP5OwdkM'
project_id = '392ac0d9e33b4e7aa394db790e5a267b'
region_name = 'cn-north-4'
model_location = "/dmetasoul-resource-bucket/jiangnan/model/"
image_path = "swr.cn-north-4.myhuaweicloud.com/dmetasoul-repo/metaspore-sagemaker-release:v1.0.34"
initial_config = dict(protocol="https", port="8080")
specification = "modelarts.vm.cpu.2u"

envs = {
    "AWS_ACCESS_KEY_ID": access_key,
    "AWS_SECRET_ACCESS_KEY": secret_key,
    "OBS_ENDPOINT": "obs.cn-north-4.myhuaweicloud.com",
    "OBS_MODEL_URL": "obs://dmetasoul-resource-bucket/jiangnan/model/bigdata-flow-sagemaker-test-20221219-005508.tar.gz"
}

session = Session(access_key=access_key,
                  secret_key=secret_key,
                  project_id=project_id,
                  region_name=region_name)

model_instance = Model(
    session,
    model_name="test_image_with_obs_sdk",  # 模型名称
    runtime=image_path,  # 自定义镜像路径
    model_version="0.1.0",  # 模型版本
    source_location_type="OBS_SOURCE",  # 模型文件来源类型
    source_location=model_location,  # 模型文件路径
    model_type="Custom",  # 模型类型
    initial_config=initial_config)
configs = [ServiceConfig(model_id=model_instance.model_id, weight="100", specification=specification, instance_count=1,
                         envs=envs)]
model_instance.deploy_predictor(configs=configs)
