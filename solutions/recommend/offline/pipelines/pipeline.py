import yaml
from .nodes import PipelineNode

class Pipeline(object):
    def __init__(self, conf_path):
        self._nodes = []
        self._conf = dict()
        with open(conf_path, 'r') as stream:
            self._conf = yaml.load(stream, Loader=yaml.FullLoader)
            print('Debug -- load config: ', self._conf)
    
    def add_node(self, node):
        if not isinstance(node, PipelineNode):
            raise TypeError(f"node must be PipelineNode; {node!r} is invalid")
        self._nodes.append(node)
        
    def run(self):
        payload = {'conf': self._conf}
        for node in self._nodes:
            payload = node.preprocess(**payload)
            payload = node(**payload)
            payload = node.postprocess(**payload)

