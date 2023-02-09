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

from abc import ABC, abstractmethod
from typing import Dict, List

import networkx as nx

from metasporeflow.offline.task.task import Task


class Scheduler(ABC):
    def __init__(self, resources, scheduler_conf, tasks: Dict[str, Task]):
        self.name = scheduler_conf.name
        self.type = scheduler_conf.kind
        self.cronExpr = scheduler_conf.data.cronExpr
        self._resources = resources
        self._scheduler_conf = scheduler_conf
        self._dag = self._get_dag(scheduler_conf.data.dag)
        self._dag_tasks: List[Task] = self._get_dag_tasks(tasks)

    @abstractmethod
    def publish(self):
        raise NotImplementedError

    def destroy(self):
        pass

    def _get_dag(self, dag_conf):
        tuples = []
        for k, value_list in dag_conf.items():
            for v in value_list:
                tuples.append((k, v))
        dag = nx.DiGraph()
        dag.add_edges_from(tuples)
        if not self._is_directed_acyclic_graph(dag):
            raise Exception(
                "%s dag is not a directed acyclic graph" % self.name)
        return dag

    def _is_directed_acyclic_graph(self, dag):
        return nx.is_directed_acyclic_graph(dag)

    def _get_dag_tasks(self, tasks: Dict[str, Task]) -> List[Task]:
        dag_tasks = []
        dag_sort_list = list(nx.topological_sort(self._dag))
        for task in dag_sort_list:
            dag_tasks.append(tasks[task])
        return dag_tasks
