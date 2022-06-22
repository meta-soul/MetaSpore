from abc import ABC, abstractmethod

class PipelineNode(ABC):
    def preprocess(self, **payload) -> dict:
        return payload
    
    def postprocess(self, **payload) -> dict:
        return payload
    
    @abstractmethod
    def __call__(self, **payload) -> dict:
        pass