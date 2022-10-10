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

class Flow:
    @classmethod
    def _add_subargparser(cls, subparsers):
        command_parser = subparsers.add_parser('flow', help='metaspore flow management')
        command_parser.set_defaults(command_executor=cls._execute_flow)
        subcommand_parsers = command_parser.add_subparsers(dest='subcommand_name')
        up_parser = subcommand_parsers.add_parser('up', help='start metaspore flow')
        up_parser.set_defaults(subcommand_executor=cls._execute_flow_up)
        down_parser = subcommand_parsers.add_parser('down', help='stop metaspore flow')
        down_parser.set_defaults(subcommand_executor=cls._execute_flow_down)
        status_parser = subcommand_parsers.add_parser('status', help='show metaspore flow status')
        status_parser.set_defaults(subcommand_executor=cls._execute_flow_status)
        reload_parser = subcommand_parsers.add_parser('reload', help='reload metaspore flow')
        reload_parser.set_defaults(subcommand_executor=cls._execute_flow_reload)

    @classmethod
    def _execute_flow(cls, args):
        print('no flow command specified')

    @classmethod
    def _execute_flow_up(cls, args):
        import asyncio
        flow_executor = cls._get_flow_executor(args)
        asyncio.run(flow_executor.execute_up())

    @classmethod
    def _execute_flow_down(cls, args):
        import asyncio
        flow_executor = cls._get_flow_executor(args)
        asyncio.run(flow_executor.execute_down())

    @classmethod
    def _execute_flow_status(cls, args):
        import asyncio
        flow_executor = cls._get_flow_executor(args)
        asyncio.run(flow_executor.execute_status())

    @classmethod
    def _execute_flow_reload(cls, args):
        import asyncio
        flow_executor = cls._get_flow_executor(args)
        asyncio.run(flow_executor.execute_reload())

    @classmethod
    def _get_flow_executor(cls, args):
        from metasporeflow.flows.flow_loader import FlowLoader
        from metasporeflow.executors.local_flow_executor import LocalFlowExecutor
        flow_loader = FlowLoader()
        resources = flow_loader.load()
        flow_executor = LocalFlowExecutor(resources)
        return flow_executor
