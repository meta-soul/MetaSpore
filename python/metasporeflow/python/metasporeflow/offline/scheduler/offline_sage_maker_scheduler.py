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

import subprocess
from .scheduler import Scheduler
from .sage_maker_entrypoint_generator import SageMakerEntrypointGenerator
from ..utils.file_util import FileUtil

class OfflineSageMakerScheduler(Scheduler):
    def __init__(self, resources, scheduler_conf, tasks):
        super().__init__(resources, scheduler_conf, tasks)

    def publish(self):
        self._upload_config()
        self._save_sage_maker_config()
        self._install_crontab()
        # TODO: cf: run the first job

    def destroy(self):
        self._uninstall_crontab()
        self._clear_sage_maker_config()
        self._clear_config()

    @property
    def _scene_name(self):
        from metasporeflow.flows.metaspore_flow import MetaSporeFlow
        flow_resource = self._resources.find_by_type(MetaSporeFlow)
        scene_name = flow_resource.name
        return scene_name

    @property
    def _s3_work_dir(self):
        from metasporeflow.flows.sage_maker_config import SageMakerConfig
        sage_maker_resource = self._resources.find_by_type(SageMakerConfig)
        s3_work_dir = sage_maker_resource.data.s3WorkDir
        return s3_work_dir

    @property
    def _crontab_expr(self):
        crontab_expr = self._scheduler_conf.data.cronExpr
        return crontab_expr

    @property
    def _crontab_command(self):
        import sys
        module_name = 'metasporeflow.runners.crontab_sage_maker_runner'
        scene_name = self._scene_name
        python = sys.executable
        crontab_command = '%s -m %s --scene %s' % (python, module_name, scene_name)
        return crontab_command

    @property
    def _crontab_entry(self):
        crontab_expr = self._crontab_expr
        crontab_command = self._crontab_command
        crontab_entry = '%s %s' % (crontab_expr, crontab_command)
        return crontab_entry

    @property
    def _local_config_dir_path(self):
        import os
        config_dir = self._scheduler_conf.data.configDir
        if config_dir is None:
            from metasporeflow.flows.metaspore_flow import MetaSporeFlow
            flow_resource = self._resources.find_by_type(MetaSporeFlow)
            resource_path = flow_resource.path
            resource_dir = os.path.dirname(resource_path)
            config_dir = os.path.join(resource_dir, 'volumes')
        return config_dir

    @property
    def _s3_config_dir_path(self):
        import os
        scene_name = self._scene_name
        s3_work_dir = self._s3_work_dir
        flow_dir = os.path.join(s3_work_dir, 'flow')
        config_dir = os.path.join(flow_dir, 'scene', scene_name)
        return config_dir

    @property
    def _sage_maker_config_path(self):
        import os
        home_dir = os.path.expanduser('~')
        flow_dir = os.path.join(home_dir, '.metaspore', 'flow')
        scene_dir = os.path.join(flow_dir, 'scene', self._scene_name)
        config_path = os.path.join(scene_dir, 'sage_maker_config.yml')
        return config_path

    def _ensure_trailing_slash(self, path):
        if path.endswith('/'):
            return path
        else:
            return path + '/'

    def _generate_entrypoint(self, s3_config_dir_path):
        import os
        import subprocess
        generator = SageMakerEntrypointGenerator(self._dag_tasks)
        text = generator.generate_entrypoint()
        s3_path = os.path.join(s3_config_dir_path, 'custom_entrypoint.sh')
        print('Generate SageMaker entrypoint to %s ...' % s3_path)
        args = ['aws', 's3', 'cp', '-', s3_path]
        subprocess.run(args, input=text.encode('utf-8'), check=True)

    def _upload_config(self):
        import subprocess
        local_path = self._ensure_trailing_slash(self._local_config_dir_path)
        s3_path = self._ensure_trailing_slash(self._s3_config_dir_path)
        print('Upload algorithm config to %s ...' % s3_path)
        args = ['aws', 's3', 'sync', '--delete', local_path, s3_path]
        subprocess.check_call(args)
        self._generate_entrypoint(s3_path)

    def _clear_config(self):
        s3_path = self._ensure_trailing_slash(self._s3_config_dir_path)
        print('Clear algorithm config %s ...' % s3_path)
        args = ['aws', 's3', 'rm', '--recursive', s3_path]
        subprocess.check_call(args)

    def _get_sage_maker_config_yaml(self):
        import cattrs
        import yaml
        from metasporeflow.flows.sage_maker_config import SageMakerConfig
        sage_maker_resource = self._resources.find_by_type(SageMakerConfig)
        sage_maker_config = dict(
            apiVersion='metaspore/v1',
            kind='SageMakerConfig',
            metadata=dict(name=sage_maker_resource.name),
            spec=cattrs.unstructure(sage_maker_resource.data),
        )
        yaml_text = yaml.dump(sage_maker_config, sort_keys=False)
        return yaml_text

    def _save_sage_maker_config(self):
        import io
        import os
        config_path = self._sage_maker_config_path
        config_dir = os.path.dirname(config_path)
        if not os.path.isdir(config_dir):
            os.makedirs(config_dir)
        config_yaml = self._get_sage_maker_config_yaml()
        print('Save SageMaker config to %s ...' % config_path)
        with io.open(config_path, 'w') as fout:
            print(config_yaml, file=fout, end='')

    def _clear_sage_maker_config(self):
        import os
        config_path = self._sage_maker_config_path
        config_dir = os.path.dirname(config_path)
        print('Clear SageMaker config %s ...' % config_path)
        if os.path.isfile(config_path):
            os.remove(config_path)
        if os.path.isdir(config_dir) and not os.listdir(config_dir):
            os.rmdir(config_dir)

    def _get_old_crontab_spec(self):
        import subprocess
        try:
            args = ['crontab', '-l']
            output = subprocess.check_output(args)
            old_spec = output.decode('utf-8')
            return old_spec
        except subprocess.CalledProcessError:
            return ''

    def _make_new_crontab_spec(self, old_spec):
        lines = old_spec.splitlines()
        command = self._crontab_command
        lines = [line for line in lines if line and not line.endswith(command)]
        crontab_entry = self._crontab_entry
        lines.append(crontab_entry)
        text = '\n'.join(lines)
        new_spec = text + '\n'
        return new_spec

    def _filter_crontab_spec(self, old_spec):
        lines = old_spec.splitlines()
        command = self._crontab_command
        lines = [line for line in lines if line and not line.endswith(command)]
        if not lines:
            return ''
        text = '\n'.join(lines)
        new_spec = text + '\n'
        return new_spec

    def _update_crontab(self, crontab_spec):
        args = ['crontab', '-']
        subprocess.run(args, input=crontab_spec.encode('utf-8'), check=True)

    def _install_crontab(self):
        old_spec = self._get_old_crontab_spec()
        new_spec = self._make_new_crontab_spec(old_spec)
        print('Install crontab entry %r ...' % self._crontab_entry)
        self._update_crontab(new_spec)

    def _uninstall_crontab(self):
        old_spec = self._get_old_crontab_spec()
        new_spec = self._filter_crontab_spec(old_spec)
        print('Uninstall crontab command %r ...' % self._crontab_command)
        self._update_crontab(new_spec)
