from attrs import frozen
from typing import Optional, Dict, List, Any


@frozen
class ResourceSpec:
    cpu: int
    memory: str


@frozen
class ResourcesSpec:
    requests: ResourceSpec
    limits: ResourceSpec


@frozen
class ContainerSpec:
    name: str
    image: str
    imagePullPolicy: str
    command: List[str]
    args: List[str]
    env: List[Any]
    resources: ResourcesSpec
    volumeMounts: List[Any]


@frozen
class TrackingMetadata:
    name: str
    namespace: str
    labels: Optional[Dict[str, str]] = None
    annotations: Optional[Dict[str, str]] = None


@frozen
class TrackingSpec:
    uploadType: str
    uploadPath: str
    accessKeyId: str
    secretAccessKey: str
    endpoint: str
    uploadWhen: str
    uploadInterval: int
    uploadBackupCount: int


@frozen
class TrackingConfig:
    apiVersion: str
    kind: str
    metadata: TrackingMetadata
    spec: TrackingSpec


class TrackingK8sGenerator():
    def __init__(self, TrackingK8sConfig):
        self._k8s_tracking_config = self._get_k8s_tracking_config()
        self.upload_type = self._k8s_tracking_config.uploadType
        self.upload_path = self._k8s_tracking_config.uploadPath
        self.access_key_id = self._k8s_tracking_config.accessKeyId
        self.secret_access_key = self._k8s_tracking_config.secretAccessKey
        self.endpoint = self._k8s_tracking_config.endpoint
        self.upload_when = self._k8s_tracking_config.uploadWhen
        self.upload_interval = self._k8s_tracking_config.uploadInterval
        self.upload_backup_count = self._k8s_tracking_config.uploadBackupCount

    def _get_job_template_metadata(self):
        job_template_metadata = TrackingMetadata(
            namespace=self._k8s_namespace,
            labels={'app': 'metasporeflow-offline'},
            annotations={'sidecar.istio.io/inject': 'false'},
        )
        return job_template_metadata

    def _get_job_template_spec(self):
        job_template_spec = TrackingSpec(
            serviceAccountName=self._service_account_name,
            restartPolicy='Never',
            containers=(
                ContainerSpec(
                    name='metasporeflow-offline-cronjob',
                    image=self._container_image,
                    imagePullPolicy='Always',
                    command=['/bin/bash', '-c', '--'],
                    args=[self._job_command],
                    env=[
                        {'name': 'AWS_ENDPOINT',
                         'valueFrom': {'secretKeyRef': {'name': 'aws-secret', 'key': 'aws_endpoint'}}},
                        {'name': 'AWS_ACCESS_KEY_ID',
                         'valueFrom': {'secretKeyRef': {'name': 'aws-secret', 'key': 'aws_access_key_id'}}},
                        {'name': 'AWS_SECRET_ACCESS_KEY',
                         'valueFrom': {'secretKeyRef': {'name': 'aws-secret', 'key': 'aws_secret_access_key'}}}
                    ],
                    resources=ResourcesSpec(
                        requests=ResourceSpec(cpu=2, memory='4Gi'),
                        limits=ResourceSpec(cpu=2, memory='4Gi'),
                    ),
                    # volumeMounts=[
                    #     {'name': 'spark-config-volume',
                    #      'mountPath': '/opt/spark/conf'},
                    #     {'name': 'aws-config-volume',
                    #      'mountPath': '/home/spark/.aws'},
                    # ],
                ),
            ),
            volumes=[
                # {'name': 'spark-config-volume',
                #  'configMap': {
                #      'name': 'spark-configmap',
                #      'items': [
                #          {'key': 'spark-defaults.conf', 'path': 'spark-defaults.conf'},
                #          {'key': 'driver-podTemplate.yaml', 'path': 'driver-podTemplate.yaml'},
                #          {'key': 'executor-podTemplate.yaml', 'path': 'executor-podTemplate.yaml'},
                #      ]}},
                # {'name': 'aws-config-volume',
                #  'configMap': {
                #      'name': 'spark-configmap',
                #      'items': [
                #          {'key': 'config', 'path': 'config'},
                #      ]}},
            ],
        )
        # if self._scheduler_conf.data.sharedConfigVolume is not None:
        #     conf = self._scheduler_conf.data.sharedConfigVolume
        #     keys = self._get_configmap_keys(self._k8s_namespace, conf.configmap)
        #     job_template_spec.containers[0].volumeMounts.append(
        #         {'name': conf.name,
        #          'mountPath': conf.mountPath})
        #     job_template_spec.volumes.append(
        #         {'name': conf.name,
        #          'configMap': {
        #              'name': conf.configmap,
        #              'items': [
        #                  {'key': key, 'path': key}
        #                  for key in keys
        #              ]}})
        return job_template_spec

    def _convert_to_yaml_text(self, conf):
        import cattrs
        import yaml
        data = cattrs.unstructure(conf)
        text = yaml.dump(data, sort_keys=False)
        return text