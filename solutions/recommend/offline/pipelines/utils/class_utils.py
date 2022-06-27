import importlib

def get_class(class_dict):
    clazz = getattr(importlib.import_module(class_dict['module_name']), class_dict['class_name'])
    print('Debug - clazz: ', clazz)
    return clazz

def get_class(module_name, class_name):
    clazz = getattr(importlib.import_module(module_name), class_name)
    print('Debug - clazz: ', clazz)
    return clazz