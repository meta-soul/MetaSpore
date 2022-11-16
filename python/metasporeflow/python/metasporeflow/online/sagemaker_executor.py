import decimal
import json

from metasporeflow.online.common import dictToObj

from sagemaker import get_execution_role, Session, image_uris
import boto3
import datetime
import time
import os
import os
import tarfile
import asyncio

from metasporeflow.online.online_generator import OnlineGenerator

class DateEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime.date):
            return obj.strftime("%Y-%m-%d")
        elif isinstance(obj, datetime.datetime):
            return obj.strftime("%Y-%m-%d %H:%M:%S")
        elif isinstance(obj, decimal.Decimal):
            return float(obj)
        else:
            return json.JSONEncoder.default(self, obj)

class SageMakerExecutor(object):
    def __init__(self, resources):
        self._online_resource = resources.find_by_name("online_local_flow")
        self._generator = OnlineGenerator(resource=self._online_resource)
        self.configure = self._online_resource.data
        self.sagemaker_info = dictToObj(self.configure.sagemaker_info)
        self.region = "cn-northwest-1" if self.sagemaker_info.region else self.sagemaker_info.region
        self.sagemaker_session = Session(boto3.session.Session(region_name=self.region))
        self.role = get_execution_role(sagemaker_session=self.sagemaker_session)
        self.sm_client = boto3.client("sagemaker", self.region)
        self.runtime_sm_client = boto3.client("runtime.sagemaker", self.region)
        self.account_id = boto3.client("sts", self.region).get_caller_identity()["Account"]
        self.bucket = "dmetasoul-test-bucket" if self.sagemaker_info.bucket else self.sagemaker_info.bucket
        self.prefix = "demo-metaspore-endpoint" if self.sagemaker_info.prefix else self.sagemaker_info.prefix

    async def create_model(self, model_name, scene, key):
        model_url = "s3://{}/{}".format(self.bucket, key)
        if self.sagemaker_info.image:
            container = self.sagemaker_info.image
        else:
            version = "v1.0.1" if self.sagemaker_info.version else self.sagemaker_info.version
            container = "{}.dkr.ecr.{}.amazonaws.com.cn/{}:{}".format(
                self.account_id, self.region, "dmetasoul-repo/metaspore-sagemaker-release", version
            )
        environment = dict()
        environment["CONSUL_ENABLE"] = "false"
        environment["SERVICE_PORT"] = "8080"
        if self.sagemaker_info.options:
            if "mongo_service" in self.sagemaker_info.options:
                environment["MONGO_HOST"] = str(self.sagemaker_info.options["mongo_service"])
            if "mongo_port" in self.sagemaker_info.options:
                environment["MONGO_PORT"] = str(self.sagemaker_info.options["mongo_port"])
        container = {"Image": container, "ModelDataUrl": model_url, "Environment": environment}
        if self.sagemaker_info.vpcSecurityGroupIds and self.sagemaker_info.vpcSubnets:
            vpc_config = {
                'SecurityGroupIds': [self.sagemaker_info.vpcSecurityGroupIds,],
                'Subnets': self.sagemaker_info.vpcSubnets
            }
            create_model_response = self.sm_client.create_model(
                ModelName=model_name,
                ExecutionRoleArn=self.role,
                Containers=[container],
                VpcConfig=vpc_config
            )
        else:
            create_model_response = self.sm_client.create_model(
                ModelName=model_name,
                ExecutionRoleArn=self.role,
                Containers=[container],
            )
        print("model resp:", create_model_response)
        print("Model Arn: " + create_model_response["ModelArn"])
        resp = self.sm_client.describe_model(ModelName=model_name)
        print("model: {} resp: ".format(model_name) + json.dumps(resp, cls=DateEncoder))
        if not scene:
            endpoint_name = "endpoint-{}".format(model_name)
        else:
            endpoint_name = scene
        print("model_url=", model_url)
        self.create_endpoint_config(endpoint_name, model_name)

    def create_endpoint_config(self, endpoint_name, model_name):
        endpoint_config_name = "config-%s" % endpoint_name
        create_endpoint_config_response = self.sm_client.create_endpoint_config(
            EndpointConfigName=endpoint_config_name,
            ProductionVariants=[
                {
                    "InstanceType": "ml.m4.xlarge",
                    "InitialInstanceCount": 1,
                    "InitialVariantWeight": 1,
                    "ModelName": model_name,
                    "VariantName": "variant-name-1",
                }
            ],
        )
        print("Endpoint config Arn: " + create_endpoint_config_response["EndpointConfigArn"])
        resp = self.sm_client.describe_endpoint_config(EndpointConfigName=endpoint_config_name)
        print("endpoint config: {} resp: ".format(endpoint_config_name) + json.dumps(resp, cls=DateEncoder))
        create_endpoint_response = self.sm_client.create_endpoint(
            EndpointName=endpoint_name, EndpointConfigName=endpoint_config_name
        )
        print("Endpoint Arn: " + create_endpoint_response["EndpointArn"])

        resp = self.sm_client.describe_endpoint(EndpointName=endpoint_name)
        status = resp["EndpointStatus"]
        print("Endpoint Status: " + status)
        print("Waiting for {} endpoint to be in service...".format(endpoint_name))
        waiter = self.sm_client.get_waiter("endpoint_in_service")
        waiter.wait(EndpointName=endpoint_name)

    def invoke_endpoint(self, endpoint_name, request):
        resp = self.sm_client.describe_endpoint(EndpointName=endpoint_name)
        status = resp["EndpointStatus"]
        print("Endpoint:" + endpoint_name + " Status: " + status)
        print("Endpoint:" + endpoint_name + " resp: " + json.dumps(resp, cls=DateEncoder))
        if not isinstance(request, dict):
            print("request type is not match request dict")
            return None
        request_body = json.dumps(request)
        response = self.runtime_sm_client.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType="application/json",
            Accept='application/json',
            Body=request_body,
        )
        res = json.loads(response["Body"].read())
        return res

    def add_model_to_s3(self, scene, model_path):
        s3 = boto3.resource('s3', self.region)
        model = self.process_model_info(scene, model_path)
        key = os.path.join(self.prefix, model)
        with open(model, "rb") as file_obj:
            s3.Bucket(self.bucket).Object(key).upload_fileobj(file_obj)
        return key

    def process_model_info(self, scene, model_paths):
        config_file = "recommend-config.yaml"
        service_confog = self._generator.gen_server_config()
        with open(config_file, "w") as file:
            file.write(service_confog)
        for model_name, model_prefix in model_paths.items():
            model_path = "model/{}".format(model_name)
            if not os.path.exists(model_path):
                os.makedirs(model_path)
            bucket = self.bucket
            if model_prefix.startswith("s3://"):
                idx = model_prefix.find("/", len("s3://"))
                if idx == -1:
                    print("model: {} path: {} is error!".format(model_name, model_prefix))
                    continue
                bucket = model_prefix[len("s3://"):idx]
                model_prefix = model_prefix[idx + 1:]
            self.download_directory(bucket, model_prefix, model_path)
        tar_file = "{}.tar.gz".format(scene)
        with tarfile.open(tar_file, "w:gz") as tar:
            tar.add(config_file)
            tar.add("model")
            tar.add("model-infos.json")
        return tar_file

    def download_directory(self, bucket_name, path, local_path):
        if os.path.isfile(local_path):
            os.remove(local_path)
        if not os.path.exists(local_path):
            os.mkdir(local_path)
        s3 = boto3.resource('s3', self.region)
        bucket = s3.Bucket(bucket_name)
        for obj in bucket.objects.filter(Prefix=path):
            local_file = os.path.join(local_path, obj.key[len(path):].lstrip("/"))
            if not os.path.exists(os.path.dirname(local_file)):
                os.makedirs(os.path.dirname(local_file))
            key = obj.key
            print(f'Downloading {key}')
            bucket.download_file(key, local_file)

    def execute_up(self, **kwargs):
        model_path = kwargs.get("models", {})
        service_confog = self._generator.gen_service_config()
        scenes = service_confog.recommend_service.scenes
        if not scenes:
            print("no scene is not config in flow config!")
            scene_name = "recommend-service"
        else:
            scene_name = scenes[0].name
        model_data_path = self.add_model_to_s3(scene_name, model_path)
        asyncio.run(self.create_model("model-{}".format(scene_name), scene_name, model_data_path))

    def execute_down(self, **kwargs):
        service_confog = self._generator.gen_service_config()
        scenes = service_confog.recommend_service.scenes
        if not scenes:
            print("no scene is not config in flow config!")
            scene_name = "recommend-service"
        else:
            scene_name = scenes[0].name
        try:
            self.sm_client.delete_endpoint(EndpointName=scene_name)
            self.sm_client.delete_endpoint_config(EndpointConfigName="config-%s" % scene_name)
            self.sm_client.delete_model(ModelName="model-{}".format(scene_name))
        except:
            print("the model or config or endpoint is not exist! or endpoint is creating")

    def execute_status(self, **kwargs):
        service_confog = self._generator.gen_service_config()
        scenes = service_confog.recommend_service.scenes
        if not scenes:
            print("no scene is not config in flow config!")
            scene_name = "recommend-service"
        else:
            scene_name = scenes[0].name
        resp = self.sm_client.describe_endpoint(EndpointName=scene_name)
        status = resp["EndpointStatus"]
        info = {"status": "DOWN"}
        if status == "InService":
            info["status"] = "UP"
        resp = self.sm_client.describe_model(ModelName="model-{}".format(scene_name))
        info["model_desc"] = json.dumps(resp, cls=DateEncoder)
        resp = self.sm_client.describe_endpoint_config(EndpointConfigName="config-%s" % scene_name)
        info["config_desc"] = json.dumps(resp, cls=DateEncoder)
        resp = self.sm_client.describe_endpoint(EndpointName=scene_name)
        info["model_desc"] = json.dumps(resp, cls=DateEncoder)

    def execute_reload(self, **kwargs):
        self.execute_down(**kwargs)
        self.execute_up(**kwargs)

if __name__ == "__main__":
    from metasporeflow.flows.flow_loader import FlowLoader
    from metasporeflow.online.online_flow import OnlineFlow
    import asyncio

    flow_loader = FlowLoader()
    flow_loader._file_name = 'test/metaspore-flow.yml'
    resources = flow_loader.load()

    online_flow = resources.find_by_type(OnlineFlow)

    executor = SageMakerExecutor(resources)
    executor.execute_down()
    executor.execute_up(models={"amazonfashion-widedeep": "s3://dmetasoul-test-bucket/qinyy/test-model-watched/amazonfashion_widedeep"})
    #with open("recommend-config.yaml") as config_file:
    #    res = executor.invoke_endpoint("guess-you-like", {"operator": "updateconfig", "config": config_file.read()})
    #    print(res)
    #res = executor.invoke_endpoint("guess-you-like", {"operator": "recommend", "request": {"user_id": "A1P62PK6QVH8LV", "scene": "guess-you-like"}})
    #print(res)
    #executor.process_model_info("guess-you-like", {"amazonfashion-widedeep": "s3://dmetasoul-test-bucket/qinyy/test-model-watched/amazonfashion_widedeep"})
