import json

from sagemaker import get_execution_role, Session, image_uris
import boto3
import datetime
import time
import os
import os
import tarfile

from python.metasporeflow.python.metasporeflow.online.online_generator import OnlineGenerator


class SageMakerExecutor(object):
    def __init__(self, resources):
        self._online_resource = resources.find_by_name("online_local_flow")
        self._generator = OnlineGenerator(resource=self._online_resource)
        self.sm_client = boto3.client(service_name="sagemaker")
        self.runtime_sm_client = boto3.client(service_name="sagemaker-runtime")
        self.account_id = boto3.client("sts").get_caller_identity()["Account"]
        self.region = boto3.Session().region_name
        self.bucket = "dmetasoul-test-bucket"
        self.prefix = "demo-multimodel-endpoint"
        self.role = get_execution_role()

    async def create_model(self, model_name, endpoint_name=None):
        model_url = "https://s3-{}.amazonaws.com/{}/{}/".format(self.region, self.bucket, self.prefix)
        container = "{}.dkr.ecr.{}.amazonaws.com/{}:latest".format(
            self.account_id, self.region, "dmetasoul-repo/metaspore-sagemaker-release"
        )

        container = {"Image": container, "ModelDataUrl": model_url, "Mode": "MultiModel"}

        create_model_response = self.sm_client.create_model(
            ModelName=model_name, ExecutionRoleArn=self.role, Containers=[container]
        )
        print("Model Arn: " + create_model_response["ModelArn"])
        if not endpoint_name:
            endpoint_name = "endpoint-{}".format(model_name)
        self.create_endpoint_config(endpoint_name, model_name)

    def create_endpoint_config(self, endpoint_name, model_name):
        endpoint_config_name = "config_%s_%s" % (model_name, endpoint_name)
        create_endpoint_config_response = self.sm_client.create_endpoint_config(
            EndpointConfigName=endpoint_config_name,
            ProductionVariants=[
                {
                    "InstanceType": "ml.m5.xlarge",
                    "InitialInstanceCount": 2,
                    "InitialVariantWeight": 1,
                    "ModelName": model_name,
                    "VariantName": "AllTraffic",
                }
            ],
        )
        print("Endpoint config Arn: " + create_endpoint_config_response["EndpointConfigArn"])
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
        if isinstance(request, dict):
            request = [request, ]
        elif not isinstance(request, list):
            print("request type is not match request list or dict")
            return []
        response = self.runtime_sm_client.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType="application/x-image",
            TargetModel="resnet_18.tar.gz",  # this is the rest of the S3 path where the model artifacts are located
            Body=request,
        )
        res = json.loads(response["Body"].read())
        print(*res, sep="\n")
        return res

    def add_model_to_s3(self, models):
        s3 = boto3.resource("s3")
        try:
            s3.meta.client.head_bucket(Bucket=self.bucket)
        except ClientError:
            s3.create_bucket(Bucket=self.bucket, CreateBucketConfiguration={"LocationConstraint": self.region})
        for model_name, s3_path in models:
            s3_bucket = s3_path.get("bucket", self.bucket)
            s3_prefix = s3_path.get("prefix")
            if not s3_prefix:
                print("s3 prefix is empty at model: {}".format(model_name))
                continue
            model = self.process_model_info(model_name, s3_bucket, s3_prefix)
            key = os.path.join(self.prefix, model)
            with open(model, "rb") as file_obj:
                s3.Bucket(self.bucket).Object(key).upload_fileobj(file_obj)

    def process_model_info(self, model_name, model_bucket, model_prefix):
        service_confog = self._generator.gen_service_config()
        with open("data/{}/service-config.yaml".format(model_name), "w") as file:
            file.write(service_confog)
        model_path = "data/models"
        self.download_directory(model_bucket, model_prefix, model_path)
        tar_file = "data/{}.gz".format(model_name)
        with tarfile.open(tar_file, "w:gz") as tar:
            tar.add("{}/{}".format(model_path, model_name), arcname=".")
        return tar_file

    def download_directory(self, bucket_name, path, local_path):
        if os.path.isfile(local_path):
            os.remove(local_path)
        if not os.path.exists(local_path):
            os.mkdir(local_path)
        s3 = boto3.resource('s3')
        bucket = s3.Bucket(bucket_name)
        for obj in bucket.objects.filter(Prefix=path):
            local_file = os.path.join(local_path, obj.key)
            if not os.path.exists(os.path.dirname(local_file)):
                os.makedirs(os.path.dirname(local_file))
            key = obj.key
            print(f'Downloading {key}')
            bucket.download_file(key, local_file)


if __name__ == "__main__":
    from metasporeflow.flows.flow_loader import FlowLoader
    from metasporeflow.online.online_flow import OnlineFlow
    import asyncio

    flow_loader = FlowLoader()
    flow_loader._file_name = 'test/metaspore-flow.yml'
    resources = flow_loader.load()

    online_flow = resources.find_by_type(OnlineFlow)
    print(type(online_flow))
    print(online_flow)

    executor = SageMakerExecutor(resources)
    executor.add_model_to_s3({
        "amazonfashion_widedeep": {
            "bucket": "dmetasoul-test-bucket",
            "prefix": "qinyy/test-model-watched/amazonfashion_widedeep"
        }
    })
    executor.create_model("amazonfashion_widedeep", "recommend")
    res = executor.invoke_endpoint("recommend", {"operator": "recommend", "user_id": "A1P62PK6QVH8LV"})
    # print(res)
