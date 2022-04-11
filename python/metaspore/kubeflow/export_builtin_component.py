import os
import kfp
import sys
# using a loop to check the /metaspore/algos/components
for module in os.listdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../algos/components')):
    # just select the .py file, eg: WideDeep.py
    if module == '__init__.py' or module[-3:] != '.py':
        continue
    # import that module, eg: WideDeep
    sys.path.append('./metaspore/algos/components')
    target_module = __import__(module[:-3], locals(), globals())
    # check all the func in that module
    for func_name in dir(target_module):
        func = getattr(target_module, func_name)
        # check if it is decorated
        if hasattr(func,'is_decorated'):
            # use the .component_name attributes
            kfp.components.func_to_container_op(
                func = func,
                base_image = 'hub.kce.ksyun.com/dmetasoul/spark:dmetasoul-v1.2.2-test',
                output_component_file= 'kubeflow_components/' + func.component_name + '.yaml',
                use_code_pickling= True
            )
del module