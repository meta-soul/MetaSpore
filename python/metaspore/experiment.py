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

import json
import time
from typing import Dict, Any

from . import patching_pickle
import os
import requests
import glob
import pickle
import shutil
import logging
import sys

SUCCESS = 'SUCCESS'
FAILED = 'FAILED'
BACKFILL = 'backfill'
ONLINE = 'online'
SUBMIT_DIR = {BACKFILL: 'backfill_dir', ONLINE: 'online_dir'}

logging.basicConfig(format='[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s',
                    stream=sys.stdout)
logger = logging.getLogger(__name__)


class Experiment(object):
    def __init__(self,
                 job_name,
                 experiment_name,
                 business_name,
                 owner,
                 schedule_interval,
                 func,
                 start_date,
                 end_date=None,
                 upstream_job_names=None,
                 extra_dag_conf: Dict[str, Any] = {},
                 enable_auth_token=True,
                 is_local_test=False,
                 delay_backfill=False,
                 airflow_host=None,
                 debug=False
                 ):
        self.job_name = job_name
        self.experiment_name = experiment_name
        self.business_name = business_name
        self.owner = owner
        self.schedule_interval = schedule_interval
        self.func = func
        self.start_date = start_date
        self.end_date = end_date
        self.upstream_job_names = upstream_job_names
        self.extra_dag_conf = extra_dag_conf
        self.enable_auth_token = enable_auth_token
        self.is_local_test = is_local_test
        self.delay_backfill = delay_backfill
        self.airflow_host = airflow_host
        self._set_logging_level(debug)

    def submit_backfill(self):
        self._run(BACKFILL)

    def submit_online(self):
        self.end_date = None
        self._run(ONLINE)

    def _run(self, submit_type):
        experiment_obj = self
        experiment_operator = ExperimentOperate(self.enable_auth_token, self.is_local_test, self.airflow_host)
        job_obj = Job(experiment_obj, experiment_operator, submit_type)
        logger.debug(self._print_attr(experiment_obj))
        logger.debug(self._print_attr(experiment_operator))
        logger.debug(self._print_attr(job_obj))
        if submit_type == ONLINE:
            experiment_operator.check_exist_dag_conf(job_obj)
        self._start_job(experiment_operator, job_obj)

    def _start_job(self, experiment_operator, job_obj):
        logger.info("start job")
        if not experiment_operator.is_local_test:
            if experiment_operator.dump_pickle(job_obj) == SUCCESS:
                experiment_operator.upload_file_to_s3(job_obj)
            else:
                raise RuntimeError("dump pickle error")
        else:
            experiment_operator.dump_pickle(job_obj)

    def _set_logging_level(self, debug):
        LOG_LEVEL = 'DEBUG' if debug else 'INFO'
        logger.setLevel(LOG_LEVEL)
        logger.info(f"LOG_LEVEL: {LOG_LEVEL}")

    def _print_attr(self, obj):
        import inspect
        class_name = str(obj.__class__)
        attr_str = [class_name]
        for i in inspect.getmembers(obj):
            # Ignores anything starting with underscore
            # (that is, private and protected attributes)
            if not i[0].startswith('_'):
                # Ignores methods
                if not inspect.ismethod(i[1]):
                    attr_str.append(str(i))
        attr_str.append('\n')
        return '\n'.join(attr_str)


class ExperimentOperate(object):
    _LOCAL_AIRFLOW_HOST = 'http://localhost:8080'
    _LOCAL_AIRFLOW_PICKLE_TMP_DIR = '/opt/airflow/dags/s3-sync'
    _LOCAL_JUPYTER_PICKLE_TMP_DIR = '.experiment_sdk/pickle_dir'
    _AIRFLOW_HOST_ENV_KEY = 'AIRFLOW_HOST'
    _CONSUL_HOST_ENV_KEY = 'CONSUL_HOST'
    _AIRFLOW_S3_SYNC_PATH_ENV_KEY = 'AIRFLOW_S3_SYNC_PATH'
    _AIRFLOW_REST_AUTHORIZATION_TOKEN = 'AIRFLOW_REST_AUTHORIZATION_TOKEN'

    def __init__(self, enable_auth_token, is_local_test, customer_airflow_host):

        self.enable_auth_token = enable_auth_token
        self.is_local_test = is_local_test
        self.airflow_host = self.get_airflow_hosts(customer_airflow_host)  # from env
        self.local_pickle_tmp_dir = self.get_local_pickle_tmp_dir()
        self.airflow_s3_sync_path = self.get_airflow_s3_sync_path()  # from env
        self.authorization_token = self.get_airflow_rest_authorization_token()

    def print_airflow_web_hosts(self):
        logger.info(f"airflow web url: {self.airflow_host}")

    @staticmethod
    def sync_from_s3_to_local(s3_path, local_path):
        try:
            shutil.rmtree(local_path)
            os.makedirs(local_path)
            result = os.system(f"aws s3 sync {s3_path} {local_path}")
        except Exception as err:
            raise RuntimeError(f"s3 sync error: {err}")
        return result

    @staticmethod
    def load_pickle_file(pickler_path):
        job_instance = None
        with open(pickler_path, 'rb') as f:
            try:
                job_instance = pickle.load(f)
            except Exception as err:
                raise Exception(f"load {pickler_path} err: {err}")
        return job_instance

    @staticmethod
    def dump_pickle_file(obj, pickler_path):
        with open(pickler_path, 'wb') as f:
            try:
                patching_pickle.dump(obj, f)
                return SUCCESS
            except:
                return FAILED

    def check_exist_dag_conf(self, job_obj):
        if not self.is_local_test:
            if job_obj.submit_type == ONLINE:
                s3_dir = '/'.join([self.airflow_s3_sync_path, job_obj.business, SUBMIT_DIR[ONLINE]])
                local_dir = '/'.join(
                    [self.local_pickle_tmp_dir, job_obj.business, SUBMIT_DIR[ONLINE]])
            elif job_obj.submit_type == BACKFILL:
                s3_dir = '/'.join(
                    [self.airflow_s3_sync_path, job_obj.business, SUBMIT_DIR[BACKFILL],
                     job_obj.experiment])
                local_dir = '/'.join(
                    [self.local_pickle_tmp_dir, job_obj.business, SUBMIT_DIR[BACKFILL],
                     job_obj.experiment])
            else:
                raise ValueError(f"no such submit_type: {job_obj.submit_type}")

            result = self.sync_from_s3_to_local(s3_dir, local_dir)
            if result != 0:
                raise RuntimeError("check exist dag conf error")
            tmp_local_dir = local_dir
        else:
            if job_obj.submit_type == ONLINE:
                tmp_local_dir = '/'.join(
                    [self.local_pickle_tmp_dir, job_obj.business, SUBMIT_DIR[ONLINE]])
            else:
                tmp_local_dir = '/'.join(
                    [self.local_pickle_tmp_dir, job_obj.business, SUBMIT_DIR[BACKFILL],
                     job_obj.experiment])

        if os.path.isdir(tmp_local_dir):
            files = glob.glob(tmp_local_dir + '/**/*.pickle', recursive=True)
            if len(files) != 0:
                random_file = files[0]
                random_job_obj = self.load_pickle_file(random_file)
                random_dag_conf = random_job_obj.dag_conf
                for key in random_dag_conf.keys():
                    r_v = random_dag_conf[key]
                    j_v = job_obj.dag_conf[key]
                    if r_v != j_v:
                        raise ValueError(
                            f"unexpect dag conf: {key}:{j_v}. it must be same with exist dag {job_obj.dag_id}: {key}:{r_v}")

    def dump_pickle(self, job_obj):
        local_pickle_file_path = job_obj.local_pickle_file_path
        if os.path.exists(local_pickle_file_path):
            os.remove(local_pickle_file_path)
        file_dir = os.path.split(local_pickle_file_path)[0]
        if not os.path.isdir(file_dir):
            os.makedirs(file_dir)
        return self.dump_pickle_file(job_obj, local_pickle_file_path)

    def get_consul_host(customer_consul_host):
        if not customer_consul_host:
            consul_host = os.getenv(ExperimentOperate._CONSUL_HOST_ENV_KEY, '')
        else:
            consul_host = customer_consul_host
        if not consul_host:
            raise ValueError(f"CONSUL_HOST:{consul_host} not set")
        try:
            tmp_list = consul_host.split(':')
            host = tmp_list[0]
            port = tmp_list[1].replace('/', '')
        except Exception as err:
            raise ValueError(f"consul_host parse err: {consul_host}, should be like '127.0.0.1:8500' . err:{err}")
        return host, port

    def get_airflow_hosts(self, customer_airflow_host):
        if self.is_local_test:
            airflow_host = ExperimentOperate._LOCAL_AIRFLOW_HOST
        else:
            if not customer_airflow_host:
                airflow_host = os.getenv(ExperimentOperate._AIRFLOW_HOST_ENV_KEY, '')
            else:
                airflow_host = customer_airflow_host
        return airflow_host

    def check_airflow_hosts(self, airflow_host):
        if not airflow_host:
            raise ValueError(f"AIRFLOW_HOST:{airflow_host} not set")

    def get_airflow_s3_sync_path(self):
        try:
            airflow_s3_sync_path = os.getenv(ExperimentOperate._AIRFLOW_S3_SYNC_PATH_ENV_KEY)
        except Exception as err:
            raise AttributeError(f"AIRFLOW_S3_SYNC_PATH not set in env; err: {err}")
        return airflow_s3_sync_path

    def get_local_pickle_tmp_dir(self):
        if self.is_local_test:
            local_pickle_tmp_dir = ExperimentOperate._LOCAL_AIRFLOW_PICKLE_TMP_DIR
        else:
            local_pickle_tmp_dir = ExperimentOperate._LOCAL_JUPYTER_PICKLE_TMP_DIR
        return local_pickle_tmp_dir

    def upload_file_to_s3(self, job_obj):
        local_pickle_file_path = job_obj.local_pickle_file_path
        if local_pickle_file_path.startswith(self.local_pickle_tmp_dir):
            suffix = local_pickle_file_path.replace(self.local_pickle_tmp_dir, '')
            s3_path = self.airflow_s3_sync_path + suffix
            try:
                cmd = "aws s3 cp {0} {1}".format(local_pickle_file_path, s3_path)
                os.system(cmd)
                logger.info(f"success upload pickle to s3: {s3_path}")
            except Exception as err:
                raise RuntimeError(f"s3 err: {err}")

    def get_airflow_rest_authorization_token(self):
        airflow_rest_authorization_token = os.getenv(ExperimentOperate._AIRFLOW_REST_AUTHORIZATION_TOKEN, '')
        logger.debug(f"airflow_rest_authorization_token: {airflow_rest_authorization_token}")
        if not airflow_rest_authorization_token:
            raise RuntimeError(f"airflow_rest_authorization_token env not set.")
        return airflow_rest_authorization_token

    def get_airflow_rest_authorization_token_jwt(self, user, passwd):
        authorization_token = ''
        if self.enable_auth_token:
            endpoit = f'{self.airflow_host}/api/v1/security/login'
            headers = {'Content-Type': 'application/json'}
            data = {"username": user, "password": passwd, "refresh": "true", "provider": "db"}
            response = requests.post(url=endpoit, headers=headers, data=json.dumps(data))
            response_dict = json.loads(response.text)
            access_token = response_dict['access_token']
            authorization_token = 'Bearer ' + access_token
        return authorization_token

    def request_post(self, url):
        # headers = {'Content-Type': 'application/x-www-form-urlencoded'}
        if self.enable_auth_token:
            headers = {'rest_api_plugin_http_token': self.authorization_token}
        else:
            headers = {'Content-Type': 'application/x-www-form-urlencoded'}
        response = requests.post(url, headers=headers)
        return response

    def request_post_jwt(self, url):
        # headers = {'Content-Type': 'application/x-www-form-urlencoded'}
        if self.is_local_test:
            headers = {'Content-Type': 'application/x-www-form-urlencoded'}
        else:
            headers = {'Authorization': self.authorization_token, 'Content-Type': 'application/x-www-form-urlencoded'}
        response = requests.post(url, headers=headers)
        return response

    def set_airflow_pools(self, pool_name, slot_count):
        description = f'test'
        url = f'{self.airflow_host}/admin/rest_api/api?api=pool&cmd=set&pool_name={pool_name}&slot_count={slot_count}&pool_description={description}'
        self.request_post(url)

    def list_dag(self):
        url = f'{self.airflow_host}/admin/rest_api/api?api=list_dags'
        response = self.request_post(url)
        text = json.loads(response.text)
        dags_info = text['output']['stdout']
        return dags_info

    def unpause(self, dag_id):
        url = f'{self.airflow_host}/admin/rest_api/api?api=unpause&dag_id={dag_id}'
        response = self.request_post(url)
        text = json.loads(response.text)
        result = text['output']['stdout']
        return result

    def unpause_dag(self, job_obj):
        dag_id = job_obj.dag_id
        is_unpause = False
        for i in range(5):
            time.sleep(6)
            dags_info = self.list_dag()
            if dag_id in dags_info:
                result = self.unpause(dag_id)
                logger.info(result)
                if result:
                    logger.info(f'success start DAG : {dag_id}')
                    is_unpause = True
                    break
        if not is_unpause:
            raise ValueError(f'DAG : {dag_id} does not exist. Please open airflow web and check again.')


class Job(object):

    def __init__(self, experiment_obj, experiment_operator, submit_type):
        self.submit_type = submit_type
        self.experiment_operator = experiment_operator
        self.local_pickle_tmp_dir = experiment_operator.local_pickle_tmp_dir
        self.experiment_obj = experiment_obj
        self.name = experiment_obj.job_name
        self.experiment = experiment_obj.experiment_name
        self.business = experiment_obj.business_name
        self.func = experiment_obj.func
        self.schedule_interval = experiment_obj.schedule_interval
        self.start_date = experiment_obj.start_date
        self.end_date = experiment_obj.end_date
        self.extra_dag_conf: Dict[str, Any] = experiment_obj.extra_dag_conf
        self.dag_conf = self._get_dag_conf()
        self.upstream_job_names = self._get_upstream_job_names(experiment_obj.upstream_job_names)

    def _get_dag_conf(self):
        dag_conf = {}
        dag_conf['dag_id'] = self.dag_id
        dag_conf['schedule_interval'] = self.schedule_interval
        dag_conf['owner'] = self.owner
        dag_conf['start_date'] = self.start_date
        dag_conf['end_date'] = self.end_date
        dag_conf['catchup'] = self.catchup
        dag_conf['airflow_kubernetes_operator_namespace'] = None
        dag_conf['airflow_kubernetes_operator_image'] = None
        dag_conf['airflow_kubernetes_operator_cpu'] = '1'
        dag_conf['airflow_kubernetes_operator_memory'] = '5Gi'
        dag_conf['pickle_file_s3_path'] = self._pickle_file_s3_path()
        dag_conf.update(self.extra_dag_conf)
        return dag_conf

    @property
    def pickle_file_name(self):
        return self.name + '.pickle'

    @property
    def catchup(self):
        return True if self.submit_type == BACKFILL else False

    @property
    def dag_id(self):
        if self.submit_type == BACKFILL:
            dag_id = f'{self.business}_{self.experiment}_{BACKFILL}'
        else:
            dag_id = f'{self.business}_{ONLINE}'
        return dag_id

    @property
    def owner(self):
        owner = self.experiment_obj.owner
        return owner if owner else self.experiment_obj.business

    @property
    def local_pickle_file_path(self):
        store_dir_path = '/'.join(
            [self.local_pickle_tmp_dir, self.business, SUBMIT_DIR[self.submit_type], self.experiment])
        if not os.path.isdir(store_dir_path):
            os.makedirs(store_dir_path)
        pickle_file_path = '/'.join([store_dir_path, self.pickle_file_name])
        return pickle_file_path

    def _get_upstream_job_names(self, upstream_job_names):
        upstream_job_names = self.experiment_obj.upstream_job_names
        if upstream_job_names:
            if type(upstream_job_names) != list:
                raise ValueError(
                    f"upstream_job_names should be a list. example: upstream_job_names = ['business_experiment_job1','business_experiment_job2']")
            return upstream_job_names
        else:
            return upstream_job_names

    def _pickle_file_s3_path(self):
        local_pickle_file_path = self.local_pickle_file_path
        suffix = local_pickle_file_path.replace(self.experiment_operator.local_pickle_tmp_dir, '')
        pickle_file_s3_path = self.experiment_operator.airflow_s3_sync_path + suffix
        return pickle_file_s3_path
