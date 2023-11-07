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

import io
import os
import re
import json
import subprocess

class ServingDockerImageBuilder(object):
    def __init__(self):
        self._dockerfile_dir = self._get_dockerfile_dir()
        self._dev_dockerfile_path = self._get_dev_dockerfile_path()
        self._serving_build_dockerfile_path = self._get_serving_build_dockerfile_path()
        self._serving_release_dockerfile_path = self._get_serving_release_dockerfile_path()
        self._project_root_dir = self._get_project_root_dir()
        self._pyproject_toml_path = self._get_pyproject_toml_path()
        self._docker_image_tag = self._get_docker_image_tag()
        self._image_repository = self._get_image_repository()

    def _get_dockerfile_dir(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        return dir_path

    def _get_dev_dockerfile_path(self):
        dir_path = self._dockerfile_dir
        file_path = os.path.join(dir_path, 'Dockerfile_dev')
        return file_path

    def _get_serving_build_dockerfile_path(self):
        dir_path = self._dockerfile_dir
        file_path = os.path.join(dir_path, 'Dockerfile_serving_build')
        return file_path

    def _get_serving_release_dockerfile_path(self):
        dir_path = self._dockerfile_dir
        file_path = os.path.join(dir_path, 'Dockerfile_serving_release')
        return file_path

    def _get_project_root_dir(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        dir_path = os.path.dirname(os.path.dirname(dir_path))
        return dir_path

    def _get_pyproject_toml_path(self):
        file_path = os.path.join(self._project_root_dir, 'pyproject.toml')
        return file_path

    def _get_docker_image_tag(self):
        with io.open(self._pyproject_toml_path) as fin:
            text = fin.read()
        match = re.search('^version = "(.+)"$', text, re.M)
        if match is None:
            message = "fail to read metaspore version "
            message += "from %r" % self._pyproject_toml_path
            raise RuntimeError(message)
        image_tag = 'v' + match.group(1)
        return image_tag

    def _get_image_repository(self):
        repo = os.environ.get('REPOSITORY')
        if repo is not None:
            return repo + '/'
        else:
            return ''

    @property
    def _dev_image_name(self):
        repo = self._image_repository
        variant = '-gpu' if self._enable_gpu else ''
        image_name = '%smetaspore-dev:%s%s' % (repo, self._docker_image_tag, variant)
        return image_name

    @property
    def _serving_build_image_name(self):
        repo = self._image_repository
        variant = '-gpu' if self._enable_gpu else ''
        image_name = '%smetaspore-serving-build:%s%s' % (repo, self._docker_image_tag, variant)
        return image_name

    @property
    def _serving_release_image_name(self):
        repo = self._image_repository
        mode = 'debug' if self._enable_debug else 'release'
        variant = '-gpu' if self._enable_gpu else ''
        image_name = '%smetaspore-serving-%s:%s%s' % (repo, mode, self._docker_image_tag, variant)
        return image_name

    def _parse_args(self):
        import argparse
        parser = argparse.ArgumentParser(description='build metaspore serving image')
        parser.add_argument('-f', '--force-rebuild', action='store_true',
            help='force rebuilding of the docker images')
        parser.add_argument('-g', '--enable-gpu', action='store_true',
            help='enable gpu support')
        parser.add_argument('-d', '--enable-debug', action='store_true',
            help='enable debug support')
        parser.add_argument('-t', '--tag', type=str,
            help='tag for docker images; default to %r' % self._docker_image_tag)
        args = parser.parse_args()
        self._force_rebuild = args.force_rebuild
        self._enable_gpu = args.enable_gpu
        self._enable_debug = args.enable_debug
        if args.tag is not None:
            self._docker_image_tag = args.tag

    def _docker_image_exists(self, image_name):
        args = ['docker', 'images', '--format', 'json']
        output = subprocess.check_output(args)
        for line in output.splitlines():
            image = json.loads(line)
            name = '%s:%s' % (image['Repository'], image['Tag'])
            if name == image_name:
                return True
        return False

    def _handle_proxy(self, args):
        import os
        http_proxy = os.environ.get('http_proxy')
        https_proxy = os.environ.get('https_proxy')
        no_proxy = os.environ.get('no_proxy')
        if http_proxy is not None or https_proxy is not None or no_proxy is not None:
            args += ['--network', 'host']
            if http_proxy is not None:
                args += ['--build-arg', 'http_proxy=%s' % http_proxy]
            if https_proxy is not None:
                args += ['--build-arg', 'https_proxy=%s' % https_proxy]
            if no_proxy is not None:
                args += ['--build-arg', 'no_proxy=%s' % no_proxy]

    def _log_command(self, args):
        message = ' '.join(args)
        print('\033[38;5;240m%s\033[m' % message)

    def _build_dev_image(self):
        image_name = self._dev_image_name
        if not self._docker_image_exists(image_name) or self._force_rebuild:
            args = ['env', 'DOCKER_BUILDKIT=1', 'docker', 'build']
            self._handle_proxy(args)
            args += ['--file', self._dev_dockerfile_path]
            if self._enable_gpu:
                args += ['--build-arg', 'RUNTIME=gpu']
            args += ['--tag', image_name]
            args += [self._project_root_dir]
            self._log_command(args)
            subprocess.check_call(args)

    def _build_serving_build_image(self):
        image_name = self._serving_build_image_name
        if not self._docker_image_exists(image_name) or self._force_rebuild:
            self._build_dev_image()
            args = ['env', 'DOCKER_BUILDKIT=1', 'docker', 'build']
            self._handle_proxy(args)
            args += ['--file', self._serving_build_dockerfile_path]
            args += ['--build-arg', 'DEV_IMAGE=%s' % self._dev_image_name]
            if self._enable_gpu:
                args += ['--build-arg', 'ENABLE_GPU=ON']
            args += ['--tag', image_name]
            args += [self._project_root_dir]
            self._log_command(args)
            subprocess.check_call(args)

    def _build_serving_release_image(self):
        image_name = self._serving_release_image_name
        if not self._docker_image_exists(image_name) or self._force_rebuild:
            self._build_serving_build_image()
            target_name = 'serving_debug' if self._enable_debug else 'serving_release'
            args = ['env', 'DOCKER_BUILDKIT=1', 'docker', 'build']
            self._handle_proxy(args)
            args += ['--file', self._serving_release_dockerfile_path]
            args += ['--build-arg', 'BUILD_IMAGE=%s' % self._serving_build_image_name]
            if self._enable_gpu:
                args += ['--build-arg', 'RUNTIME=gpu']
            args += ['--tag', image_name]
            args += ['--target', target_name]
            args += [self._project_root_dir]
            self._log_command(args)
            subprocess.check_call(args)

    def run(self):
        self._parse_args()
        self._build_serving_release_image()

def main():
    builder = ServingDockerImageBuilder()
    builder.run()

if __name__ == '__main__':
    main()
