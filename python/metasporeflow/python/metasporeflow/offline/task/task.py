from abc import ABC, abstractmethod


class Task(ABC):

    def __init__(self,
                 name,
                 type,
                 data
                 ):
        self._name = name
        self._type = type
        self._data = data

    def __repr__(self):
        return '%s(%s, %s, %s)' % (self.__class__.__name__,
                               self._name,
                               self._type,
                               self._data)

    @abstractmethod
    def _execute(self):
        raise NotImplementedError

    @property
    def name(self):
        return self._name

    @property
    def type(self):
        return self._type

    @property
    def data(self):
        return self._data

    @property
    def execute(self):
        return self._execute()
