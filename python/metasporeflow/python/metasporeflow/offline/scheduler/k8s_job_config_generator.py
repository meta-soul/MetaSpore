#
# Copyright 2022 DMetaSoul
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from attrs import frozen
from typing import Any
from typing import Optional
from typing import List
from typing import Dict

@frozen
class JobTemplateMetadata:
    namespace: Optional[str] = None
    labels: Optional[Dict[str, str]] = None
    annotations: Optional[Dict[str, str]] = None

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
class JobTemplateSpec:
    serviceAccountName: str
    restartPolicy: str
    containers: List[ContainerSpec]
    volumes: List[Any]

@frozen
class JobTemplate:
    metadata: JobTemplateMetadata
    spec: JobTemplateSpec

@frozen
class BatchJobMetadata:
    name: str
    namespace: str
    labels: Optional[Dict[str, str]] = None
    annotations: Optional[Dict[str, str]] = None

@frozen
class BatchJobSpec:
    template: JobTemplate

@frozen
class BatchJob:
    apiVersion: str
    kind: str
    metadata: BatchJobMetadata
    spec: BatchJobSpec

@frozen
class CronJobMetadata:
    name: str
    namespace: str
    labels: Optional[Dict[str, str]] = None
    annotations: Optional[Dict[str, str]] = None

@frozen
class CronJobSpecJobTemplate:
    spec: BatchJobSpec

@frozen
class CronJobSpec:
    schedule: str
    jobTemplate: CronJobSpecJobTemplate

@frozen
class CronJob:
    apiVersion: str
    kind: str
    metadata: CronJobMetadata
    spec: CronJobSpec

class K8sJobConfigGenerator(object):
    def __init__(self, scheduler_conf, job_command):
        from ...flows.metaspore_offline_flow import OfflineK8sCronjobScheduler
        self._scheduler_conf = scheduler_conf
        self._k8s_namespace = self._scheduler_conf.data.namespace
        self._service_account_name = self._scheduler_conf.data.serviceAccountName
        self._container_image = self._scheduler_conf.data.containerImage
        self._cronjob_schedule = self._scheduler_conf.data.cronExpr
        self._job_command = job_command

    def _get_job_template(self):
        job_template_metadata = self._get_job_template_metadata()
        job_template_spec = self._get_job_template_spec()
        job_template = JobTemplate(
            metadata=job_template_metadata,
            spec=job_template_spec,
        )
        return job_template

    def _get_job_template_metadata(self):
        job_template_metadata = JobTemplateMetadata(
            namespace=self._k8s_namespace,
            labels={'app': 'metasporeflow-offline'},
            annotations={'sidecar.istio.io/inject': 'false'},
        )
        return job_template_metadata

    def _get_job_template_spec(self):
        job_template_spec = JobTemplateSpec(
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
                        {'name': 'SPARK_LOCAL_HOSTNAME',
                         'valueFrom': {'fieldRef': {'fieldPath': 'status.podIP'}}},
                        {'name': 'AWS_ENDPOINT',
                         'valueFrom': {'secretKeyRef': {'name': 'aws-secret', 'key': 'aws_endpoint'}}},
                        {'name': 'AWS_ACCESS_KEY_ID',
                         'valueFrom': {'secretKeyRef': {'name': 'aws-secret', 'key': 'aws_access_key_id'}}},
                        {'name': 'AWS_SECRET_ACCESS_KEY',
                         'valueFrom': {'secretKeyRef': {'name': 'aws-secret', 'key': 'aws_secret_access_key'}}},
                        {'name': 'SPARK_CONF_DIR',
                         'value': '/opt/spark/conf'},
                    ],
                    resources=ResourcesSpec(
                        requests=ResourceSpec(cpu=2, memory='4Gi'),
                        limits=ResourceSpec(cpu=2, memory='4Gi'),
                    ),
                    volumeMounts=[
                        {'name': 'spark-config-volume',
                         'mountPath': '/opt/spark/conf'},
                        {'name': 'aws-config-volume',
                         'mountPath': '/home/spark/.aws'},
                    ],
                ),
            ),
            volumes=[
                {'name': 'spark-config-volume',
                 'configMap': {
                    'name': 'spark-configmap',
                    'items': [
                        {'key': 'spark-defaults.conf', 'path': 'spark-defaults.conf'},
                        {'key': 'driver-podTemplate.yaml', 'path': 'driver-podTemplate.yaml'},
                        {'key': 'executor-podTemplate.yaml', 'path': 'executor-podTemplate.yaml'},
                    ]}},
                {'name': 'aws-config-volume',
                 'configMap': {
                    'name': 'spark-configmap',
                    'items': [
                        {'key': 'config', 'path': 'config'},
                    ]}},
            ],
        )
        if self._scheduler_conf.data.sharedConfigVolume is not None:
            conf = self._scheduler_conf.data.sharedConfigVolume
            keys = self._get_configmap_keys(self._k8s_namespace, conf.configmap)
            job_template_spec.containers[0].volumeMounts.append(
                {'name': conf.name,
                 'mountPath': conf.mountPath})
            job_template_spec.volumes.append(
                {'name': conf.name,
                 'configMap': {
                    'name': conf.configmap,
                    'items': [
                        {'key': key, 'path': key}
                        for key in keys
                    ]}})
        return job_template_spec

    def _get_configmap_keys(self, k8s_namespace, configmap_name):
        import json
        import subprocess
        try:
            args = ['kubectl', 'get', 'configmap', '-n', k8s_namespace, configmap_name, '-o', 'json']
            output = subprocess.check_output(args)
        except subprocess.CalledProcessError:
            print(f'configmap {configmap_name!r} is not found')
            raise
        keys = []
        configmap = json.loads(output)
        if 'data' in configmap:
            keys.extend(configmap['data'].keys())
        if 'binaryData' in configmap:
            keys.extend(configmap['binaryData'].keys())
        return tuple(keys)

    def _convert_to_yaml_text(self, conf):
        import cattrs
        import yaml
        data = cattrs.unstructure(conf)
        text = yaml.dump(data, sort_keys=False)
        return text

    def _generate_batch_job_config(self):
        job_template = self._get_job_template()
        batch_job = BatchJob(
            apiVersion='batch/v1',
            kind='Job',
            metadata=BatchJobMetadata(
                name='metasporeflow-offline-job',
                namespace=self._k8s_namespace,
                labels={'app': 'metasporeflow-offline'},
                annotations={'sidecar.istio.io/inject': 'false'},
            ),
            spec=BatchJobSpec(
                template=job_template,
            ),
        )
        output = self._convert_to_yaml_text(batch_job)
        return output

    def _generate_cron_job_config(self):
        job_template = self._get_job_template()
        cron_job = CronJob(
            apiVersion='batch/v1',
            kind='CronJob',
            metadata=CronJobMetadata(
                name='metasporeflow-offline-cronjob',
                namespace=self._k8s_namespace,
                labels={'app': 'metasporeflow-offline'},
                annotations={'sidecar.istio.io/inject': 'false'},
            ),
            spec=CronJobSpec(
                schedule=self._cronjob_schedule,
                jobTemplate=CronJobSpecJobTemplate(
                    spec=BatchJobSpec(
                        template=job_template,
                    ),
                ),
            ),
        )
        output = self._convert_to_yaml_text(cron_job)
        return output

    def generate_job_config(self, for_cronjob=False):
        if for_cronjob:
            text = self._generate_cron_job_config()
        else:
            text = self._generate_batch_job_config()
        return text
