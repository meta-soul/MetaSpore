import kfp
def load_component(name):
    import os
    src_dir = os.path.abspath(os.path.join(__file__,'../../kubeflow_components')) 
    my_op = kfp.components.load_component_from_file(os.path.join(src_dir, name + '.yaml'))
    return my_op