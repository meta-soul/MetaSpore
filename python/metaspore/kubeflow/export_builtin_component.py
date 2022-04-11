import os
import kfp
import sys
# using a loop to check the current path
for module in os.listdir(os.getcwd() + "/metaspore/algos/components"):
    if module == '__init__.py' or module[-3:] != '.py':
        continue
    sys.path.append('./metaspore/algos/components')
    target_module = __import__(module[:-3], locals(), globals())
    for func_name in dir(target_module):
        print(func_name)
        func = getattr(target_module, func_name)
        if hasattr(func,'is_decorated'):
            kfp.components.func_to_container_op(
                func = func,
                base_image = 'hub.kce.ksyun.com/dmetasoul/spark:dmetasoul-v1.2.2-test',
                output_component_file= 'kubeflow_components/' + func.component_name + '.yaml',
                use_code_pickling= True
            )
del module