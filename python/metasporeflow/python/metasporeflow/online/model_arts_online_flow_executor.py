import decimal
import json


import datetime
import re
import os
import shutil
import tempfile
import tarfile
from obs import ObsClient
from modelarts.session import Session
from modelarts.model import Predictor
from modelarts.model import Model
from modelarts.config.model_config import ServiceConfig
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


class ModelArtsOnlineFlowExecutor(object):
    def __init__(self, resources):
        from metasporeflow.online.online_flow import OnlineFlow
        self.now_time = datetime.datetime.now()
        self.model_version = datetime.datetime.strftime(self.now_time, "%Y%m%d-%H%M%S")
        self._resources = resources
        self._online_resource = resources.find_by_type(OnlineFlow)
        self._generator = OnlineGenerator(resource=self._online_resource)
        self.server_config = self._generator.gen_server_config()
        self.configure = self._online_resource.data
        self.region = self._get_obs_region()
        self.bucket, self.prefix = self._get_bucket_and_prefix()
        self.scene_name = self._get_scene_name()
        self.session = self._get_modelarts_session()
        self.service_weight = "100"
        self.service_instance_count = 1
        self.service_specification = "modelarts.vm.cpu.2u"
        self.service_infer_type = "real-time"

    @property
    def _obs_access_key_id(self):
        model_arts_config = self._get_model_arts_config()
        access_key_id = model_arts_config.accessKeyId
        return access_key_id

    @property
    def _obs_secret_access_key(self):
        model_arts_config = self._get_model_arts_config()
        secret_access_key = model_arts_config.secretAccessKey
        return secret_access_key

    def _get_scene_name(self):
        from metasporeflow.flows.metaspore_flow import MetaSporeFlow
        flow_resource = self._resources.find_by_type(MetaSporeFlow)
        scene_name = flow_resource.name
        return scene_name

    def _get_endpoint_name(self):
        scene_name = self._get_scene_name()
        endpoint_name = re.sub('[^A-Za-z0-9]', '-', scene_name)
        return endpoint_name

    def _get_model_arts_config(self):
        from metasporeflow.flows.model_arts_config import ModelArtsConfig
        model_arts_resource = self._resources.find_by_type(ModelArtsConfig)
        model_arts_config = model_arts_resource.data
        return model_arts_config

    def _get_iam_role(self):
        config = self._get_sage_maker_config()
        role = config.roleArn
        return role

    def _get_obs_endpoint(self):
        config = self._get_model_arts_config()
        obs_endpoint = config.obsEndpoint
        return obs_endpoint

    def _get_obs_work_dir(self):
        config = self._get_model_arts_config()
        obs_work_dir = config.obsWorkDir
        return obs_work_dir

    def _get_serving_dir(self):
        obs_work_dir = self._get_obs_work_dir()
        scene_name = self._get_scene_name()
        flow_dir = os.path.join(obs_work_dir, 'flow')
        scene_dir = os.path.join(flow_dir, 'scene', scene_name)
        model_dir = os.path.join(scene_dir, 'model')
        serving_dir = os.path.join(model_dir, 'serving')
        return serving_dir

    def _get_obs_region(self):
        import re
        pattern = r'obs\.([A-Za-z0-9\-]+?)\.myhuaweicloud\.com(\.cn)?$'
        obs_endpoint = self._get_obs_endpoint()
        match = re.match(pattern, obs_endpoint)
        if match is None:
            message = 'invalid obs endpoint %r' % obs_endpoint
            raise RuntimeError(message)
        obs_region = match.group(1)
        return obs_region

    def _get_modelarts_endpoint(self):
        region = self.region
        return 'modelarts.' + region + '.myhuaweicloud.com'

    def _get_bucket_and_prefix(self):
        from urllib.parse import urlparse
        serving_dir = self._get_serving_dir()
        results = urlparse(serving_dir, allow_fragments=False)
        bucket = results.netloc
        prefix = results.path.strip('/') + '/'
        return bucket, prefix

    def _get_container_image(self):
        url = 'swr.cn-north-4.myhuaweicloud.com'
        url += '/dmetasoul-repo/metaspore-modelarts-release:v1.0.6'
        return url

    def _endpoint_exists(self, endpoint_name):
        import botocore
        try:
            _response = self.sm_client.describe_endpoint(EndpointName=endpoint_name)
            return True
        except botocore.exceptions.ClientError:
            return False

    def _wait_endpoint(self, endpoint_name):
        import time
        counter = 0
        while True:
            status = self._get_endpoint_status(endpoint_name)
            if counter > 7200:
                message = 'fail to wait endpoint %r' % endpoint_name
                raise RuntimeError(message)
            if counter % 60 == 0:
                print('Wait endpoint %r ... [%s]' % (endpoint_name, status))
            if status in ('InService', 'Failed'):
                return status
            time.sleep(1)
            counter += 1

    def _get_modelarts_session(self):
        model_arts_config = self._get_model_arts_config()
        session = Session(access_key=self._obs_access_key_id,
                          secret_key=self._obs_secret_access_key,
                          project_id=model_arts_config.projectId,
                          region_name=self.region)
        return session

    def _generate_model_location(self, model_url):
        from urllib.parse import urlparse
        results = urlparse(model_url, allow_fragments=False)
        path_list = results.path.strip("/").split("/")
        path_list.pop(-1)
        path_list.insert(0, results.netloc)
        location = "/{}/".format("/".join(path_list))
        return location

    def create_model(self, key):
        model_url = "obs://{}/{}".format(self.bucket, key)
        model_location = self._generate_model_location(model_url)
        container_image = self._get_container_image()
        initial_config = dict(protocol="https", port="8080")
        model_arts_config = self._get_model_arts_config()
        envs = {
            "AWS_ACCESS_KEY_ID": self._obs_access_key_id,
            "AWS_SECRET_ACCESS_KEY": self._obs_secret_access_key,
            "OBS_ENDPOINT": self._get_obs_endpoint(),
            "MODELARTS_ENDPOINT": self._get_modelarts_endpoint(),
            "OBS_MODEL_URL": model_url,
            "CONSUL_ENABLE": "false"
        }
        model_instance = Model(
            self.session,
            model_name=self.scene_name,
            runtime=container_image,
            source_location_type="OBS_SOURCE",
            source_location=model_location,
            model_type="Custom",
            initial_config=initial_config)
        model_id = model_instance.model_id
        configs = [
            ServiceConfig(model_id=model_id, weight=self.service_weight, specification=self.service_specification, instance_count=self.service_instance_count, envs=envs)]
        predictor_instance = model_instance.deploy_predictor(service_name=self.scene_name,
                                                             infer_type=self.service_infer_type,
                                                             pool_name=model_arts_config.resourcePoolId,
                                                             configs=configs)
        predictor_info = predictor_instance.get_service_info()
        print("service access_address. please save it for inference: " + str(predictor_info["access_address"]))

    def create_endpoint(self, endpoint_name, endpoint_config_name):
        create_endpoint_response = self.sm_client.create_endpoint(
            EndpointName=endpoint_name, EndpointConfigName=endpoint_config_name
        )
        print("Endpoint Arn: " + create_endpoint_response["EndpointArn"])

        resp = self.sm_client.describe_endpoint(EndpointName=endpoint_name)
        status = resp["EndpointStatus"]
        print("Endpoint Status: " + status)
        print("Waiting for {} endpoint to be in service...".format(endpoint_name))
        status = self._wait_endpoint(endpoint_name)
        print("Endpoint Status: " + status)
        print("{} endpoint create successfully, is in service...".format(endpoint_name))

    def update_endpoint(self, endpoint_name, endpoint_config_name):
        create_endpoint_response = self.sm_client.update_endpoint(
            EndpointName=endpoint_name, EndpointConfigName=endpoint_config_name
        )
        print("Endpoint Arn: " + create_endpoint_response["EndpointArn"])

        resp = self.sm_client.describe_endpoint(EndpointName=endpoint_name)
        status = resp["EndpointStatus"]
        print("Endpoint Status: " + status)
        print("Waiting for {} endpoint to be update in service...".format(endpoint_name))
        status = self._wait_endpoint(endpoint_name)
        print("Endpoint Status: " + status)
        print("{} endpoint update successfully is in service...".format(endpoint_name))

    def create_or_update_endpoint(self, endpoint_name, endpoint_config_name):
        if self._endpoint_exists(endpoint_name):
            self.update_endpoint(endpoint_name, endpoint_config_name)
        else:
            self.create_endpoint(endpoint_name, endpoint_config_name)

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

    def add_model_to_s3(self, endpoint_name, model_paths):
        temp_dir = tempfile.mkdtemp()
        try:
            obs_client = ObsClient(
                access_key_id=self._obs_access_key_id,
                secret_access_key=self._obs_secret_access_key,
                server=self._get_obs_endpoint(),
            )

            model = self.process_model_info(endpoint_name, model_paths, temp_dir)
            key = os.path.join(self.prefix, os.path.basename(model))
            with open(model, "rb") as file_obj:
                try:
                    resp = obs_client.putObject(self.bucket, key, file_obj)
                    if resp.status < 300:
                        print("requestId: {}. url: {}".format(resp.requestId, resp.body.url))
                    else:
                        print("errorCode: {}. errorMessage: {}".format(resp.errorCode, resp.errorMessage))
                except:
                    import traceback
                    print(traceback.format_exc())
            return key
        finally:
            shutil.rmtree(temp_dir)

    def process_model_info(self, endpoint_name, model_paths, temp_dir):
        tarball_dir = os.path.join(temp_dir, "tarball_dir")
        os.makedirs(tarball_dir)
        config_file = os.path.join(tarball_dir, "recommend-config.yaml")
        with open(config_file, "w") as file:
            file.write(self.server_config)
        model_info_file = os.path.join(tarball_dir, "model-infos.json")
        model_infos = list()
        for model_name, model_prefix in model_paths.items():
            rel_model_path = os.path.join("model", model_name)
            rt_model_path = os.path.join("/opt/ml/model", rel_model_path)
            model_path = os.path.join(tarball_dir, rel_model_path)
            if not os.path.isdir(model_path):
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
            model_info = dict(
                modelName=model_name,
                version="1",
                dirPath=rt_model_path,
                host="127.0.0.1",
                port=50000,
            )
            model_infos.append(model_info)
        with open(model_info_file, "w") as model_file:
            json.dump(model_infos, model_file)
            print(file=model_file)
        tarball_name = "%s-%s.tar.gz" % (endpoint_name, self.model_version)
        tarball_path = os.path.join(temp_dir, tarball_name)
        with tarfile.open(tarball_path, "w:gz") as tar:
            for name in os.listdir(tarball_dir):
                path = os.path.join(tarball_dir, name)
                tar.add(path, name)
        shutil.rmtree(tarball_dir)
        return tarball_path

    def download_directory(self, bucket_name, path, local_path):
        if not os.path.isdir(local_path):
            os.mkdir(local_path)
        obs_client = ObsClient(
            access_key_id=self._obs_access_key_id,
            secret_access_key=self._obs_secret_access_key,
            server=self._get_obs_endpoint(),
        )
        obs_objects = obs_client.listObjects(bucketName=bucket_name, prefix=path)
        for obj in obs_objects.body["contents"]:
            local_file = os.path.join(local_path, obj.key[len(path):].lstrip("/"))
            if not os.path.isdir(os.path.dirname(local_file)):
                os.makedirs(os.path.dirname(local_file))
            key = obj.key
            print(f'Downloading {key}')
            try:
                resp = obs_client.getObject(bucket_name, key, downloadPath=local_file)
                if resp.status < 300:
                    print("Downloading obs requestId: {}. key: {}".format(resp.requestId, key))
                else:
                    print("errorCode: {}. errorMessage: {}".format(resp.errorCode, resp.errorMessage))
            except:
                import traceback
                print(traceback.format_exc())

    def _get_model_ids(self, model_name):
        model_list = Model.get_model_list(self.session,
                                          model_status="published",
                                          model_name=model_name,
                                          order="desc")
        models_ids = [model["model_id"] for model in model_list["models"]]
        return models_ids

    def _get_service_ids(self, service_name):
        predictor_list = Predictor.get_service_list(
            self.session,
            service_name=service_name,
            order="asc",
            offset="0",
            infer_type=self.service_infer_type)
        services_ids = [service["service_id"] for service in predictor_list["services"]]
        return services_ids

    def execute_up(self, **kwargs):
        model_paths = kwargs.get("models", {})
        endpoint_name = self._get_endpoint_name()
        print("_get_scene_name: {}".format(self._get_scene_name()))
        model_data_path = self.add_model_to_s3(endpoint_name, model_paths)
        self.create_model(model_data_path)

    def execute_down(self, **kwargs):
        import time
        scene_name = self.scene_name
        models_ids = self._get_model_ids(scene_name)
        services_ids = self._get_service_ids(scene_name)
        print("models_ids: {}. services_ids: {}.".format(models_ids, services_ids))
        if len(models_ids) != 0 and len(services_ids) != 0:
            for model_id in models_ids:
                # deleting service
                for service_id in services_ids:
                    try:
                        service = Predictor(self.session, service_id=service_id)
                        print("stoppping service...\n service_name: {}. service_id: {}.".format(scene_name, service_id))
                        service_config = service.update_service_config(description="description",
                                                                       status="stopped",
                                                                       configs=[ServiceConfig(model_id=model_id,
                                                                                              weight=self.service_weight,
                                                                                              instance_count=self.service_instance_count,
                                                                                              specification=self.service_specification)])
                        print(service_config)
                        count = 0
                        while True:
                            service_info = service.get_service_info()
                            status = service_info["status"]
                            print("service status: {}.".format(status))
                            if status == "stopped":
                                print("deleting service...\n service_name: {}. service_id: {}.".format(scene_name, service_id))
                                service.delete_service()
                                break
                            else:
                                count += 1
                                time.sleep(5)
                                if count > 60:
                                    print("service status is not stopped.")
                    except Exception as e:
                        print(e)
        if len(models_ids) != 0:
            # deleting model
            print("deleting model...\n model_name: {}.".format(scene_name))
            for model_id in models_ids:
                model_instance = Model(self.session, model_id=model_id)
                model_instance.delete_model()

    def execute_status(self, **kwargs):
        service_name = self.scene_name
        infos = []
        predictor_list = Predictor.get_service_list(
            self.session, service_name=service_name, order="asc", offset="0", infer_type=self.service_infer_type)
        if predictor_list["count"] > 0:
            infos = [service["status"] for service in predictor_list["services"]]
        else:
            print("No service found: {}.".format(service_name))
        print("scene name: {}. service status: {}".format(service_name, str(infos)))
        return str(infos)

    def execute_reload(self, **kwargs):
        import time
        print("_get_scene_name: {}".format(self._get_scene_name()))
        count = 0
        while True:
            models_ids = self._get_model_ids(self.scene_name)
            if len(models_ids) != 0:
                self.execute_down()
                time.sleep(5)
                count += 1
                if count > 60:
                    print("execute_down timeout.")
                    break
            else:
                break
        model_paths = kwargs.get("models", {})
        endpoint_name = self._get_endpoint_name()
        model_data_path = self.add_model_to_s3(endpoint_name, model_paths)
        self.create_model(model_data_path)

    def execute_update(self):
        message = "execute_update is not supported by SageMaker"
        raise RuntimeError(message)


if __name__ == "__main__":
    from metasporeflow.flows.flow_loader import FlowLoader
    from metasporeflow.online.online_flow import OnlineFlow
    from metasporeflow.flows.model_arts_config import ModelArtsConfig

    flow_loader = FlowLoader()
    flow_loader._file_name = 'metasporeflow/online/test/metaspore-flow.yml'
    resources = flow_loader.load()
    resource = resources.find_by_type(ModelArtsConfig)
    modelarts_config = resource.data
    executor = ModelArtsOnlineFlowExecutor(resources)
    # # print(executor.execute_update())
    # executor.execute_down()
    # executor.execute_up(models={
    #                     "amazonfashion_widedeep": "s3://dmetasoul-resource-bucket/demo/workdir/flow/scene/bigdata_flow_modelarts_test/model/export/20230210-1909/widedeep"})
    # executor.execute_reload(models={
    #                         "amazonfashion_widedeep": "s3://dmetasoul-resource-bucket/demo/workdir/flow/scene/bigdata_flow_modelarts_test/model/export/20230210-1909/widedeep"})
    print(executor.execute_status())
    # # executor.execute_down()

    # # executor.execute_up(models={"amazonfashion_widedeep": "s3://dmetasoul-test-bucket/qinyy/test-model-watched/amazonfashion_widedeep"})
    # # with open("recommend-config.yaml") as config_file:
    # #    res = executor.invoke_endpoint("guess-you-like", {"operator": "updateconfig", "config": config_file.read()})
    # #    print(res)

    # endpoint_name = executor._get_endpoint_name()
    # res = executor.invoke_endpoint(endpoint_name, {"operator": "recommend",
    #                                                "request": {"user_id": "A1P62PK6QVH8LV", "scene": "guess-you-like"}})
    # print(res)
    # # executor.process_model_info("guess-you-like", {"amazonfashion_widedeep": "s3://dmetasoul-test-bucket/qinyy/test-model-watched/amazonfashion_widedeep"})
