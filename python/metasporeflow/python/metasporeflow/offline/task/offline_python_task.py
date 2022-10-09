from metasporeflow.offline.task.task import Task


class OfflinePythonTask(Task):

    def __init__(self,
                 name,
                 type,
                 data
                 ):
        super().__init__(name,
                         type,
                         data
                         )
        self._script_path = data.scriptPath
        self._config_path = data.configPath

    def _execute(self):
        return "python -u %s --conf %s" % (self._script_path, self._config_path)
