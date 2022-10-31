from abc import ABC, abstractmethod
from typing import Dict, List

import networkx as nx

from metasporeflow.offline.task.task import Task


class Scheduler(ABC):
    def __init__(self, scheduler_conf, tasks: Dict[str, Task]):
        self.name = scheduler_conf.name
        self.type = scheduler_conf.kind
        self.cronExpr = scheduler_conf.data.cronExpr
        self._dag = self._get_dag(scheduler_conf.data.dag)
        self._dag_tasks: List[Task] = self._get_dag_tasks(tasks)

    @abstractmethod
    def publish(self):
        raise NotImplementedError

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
