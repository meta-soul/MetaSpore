import importlib

def get_class(class_dict):
    clazz = getattr(importlib.import_module(class_dict['module_name']), class_dict['class_name'])
    print('Debug - clazz: ', clazz)
    return clazz