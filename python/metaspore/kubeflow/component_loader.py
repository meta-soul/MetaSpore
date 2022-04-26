import kfp
from ..file_utils import file_exists

def load_component(name: str, use_builin_component = True):
    import os
    default_image = 'python:3.7'
    unified_pipeline_image = os.environ['PIPELINE_IMAGE'] # give a unified image, read from local env

    if use_builin_component:
        builin_dir = os.path.abspath(os.path.join(__file__,'../../kubeflow_components')) 
        builin_file = os.path.join(builin_dir, name + '.yaml')
        # eg: ~/.local/lib/python3.8/site-packages/metaspore/kubeflow_components/WideDeep.yaml
        if file_exists(builin_file):
            my_op = kfp.components.load_component_from_file(builin_file)
            print("load builtlin component: " + name)
            my_op.component_spec.implementation.container.image = unified_pipeline_image
        else:
            message = "no such builtin component: '%s'" % name
            raise TypeError(message)
    else:
        if file_exists(name):
            try:
                my_op = kfp.components.load_component_from_file(os.path.join(name))
            except: # user input file cannot be load
                message = "your input component is incompatible"
                raise TypeError(message)

            print("load your component: " + name)
            if my_op.component_spec.implementation.container.image == default_image: # default image, eg:python3.7
                my_op.component_spec.implementation.container.image = unified_pipeline_image
        else:   # path is invalid
            message = "your input component path: '%s' is invalid " % name
            raise TypeError(message)
    return my_op