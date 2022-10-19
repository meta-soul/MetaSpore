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

class Main:
    @classmethod
    def execute(cls):
        args = cls._parse_args()
        if hasattr(args, 'subcommand_executor'):
            subcommand_executor = args.subcommand_executor
            subcommand_executor(args)
        elif hasattr(args, 'command_executor'):
            command_executor = args.command_executor
            command_executor(args)
        else:
            print('no command specified')

    @classmethod
    def _parse_args(cls):
        parser = cls._create_argparser()
        args = parser.parse_args()
        return args

    @classmethod
    def _create_argparser(cls):
        import argparse
        parser = argparse.ArgumentParser(prog='metaspore', description='metaspore flow cli')
        subparsers = parser.add_subparsers(dest='command_name')
        cls._add_subargparsers(subparsers)
        return parser

    @classmethod
    def _add_subargparsers(cls, subparsers):
        from .flow import Flow
        Flow._add_subargparser(subparsers)
