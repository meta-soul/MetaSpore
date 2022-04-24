import os
import sys
import json
import argparse
from collections import OrderedDict

import yaml

def set_basic_conf(exp_conf):
    base_conf = {}
    base_conf['experiment'] = exp_conf['experiment']
    base_conf['version'] = exp_conf['version']
    base_conf['working_dir'] = exp_conf['working_dir']
    base_conf['input_dir'] = exp_conf['input_dir']
    base_conf['output_dir'] = os.path.join(exp_conf['output_dir'], 
        exp_conf['experiment'], exp_conf['version'])
    base_conf['log_dir'] = os.path.join(exp_conf['output_dir'], 
        exp_conf['experiment'], exp_conf['version'], 'logs')
    base_conf['python'] = exp_conf['python']
    return base_conf

def set_train_eval_conf(base_conf, exp_conf):
    if 'train-eval' not in exp_conf:
        base_conf['train-eval']['status'] = 0
        return base_conf
    base_conf['train-eval']['args'].update(exp_conf['train-eval'])
    base_conf['train-eval']['args']['model_list'] = [{'name': base_conf['train']['name'], 'path': base_conf['train']['args']['model_save_dir']}]
    return base_conf

def set_train_bench_conf(base_conf, exp_conf):
    if 'train-bench' not in exp_conf:
        base_conf['train-bench']['status'] = 0
        return base_conf
    base_conf['train-bench']['args'].update(exp_conf['train-bench'])
    base_conf['train-bench']['args']['model'] = base_conf['train']['args']['model_save_dir']
    return base_conf

def set_train_conf(base_conf, exp_conf, basic_conf):
    if 'train' not in exp_conf:
        base_conf['train']['status'] = 0
        base_conf['train-eval']['status'] = 0
        base_conf['train-bench']['status'] = 0
        return base_conf
    base_conf['train']['args'].update(exp_conf['train'])
    base_conf['train']['args']['exp_name'] = basic_conf['experiment']
    base_conf['train']['args']['model_save_dir'] = 'train_{}_{}'.format(base_conf['train']['args']['task_type'], base_conf['train']['args']['loss_type'])
    base_conf = set_train_eval_conf(base_conf, exp_conf)
    base_conf = set_train_bench_conf(base_conf, exp_conf)
    return base_conf

def set_distill_eval_conf(base_conf, exp_conf):
    if 'distill-eval' not in exp_conf:
        base_conf['distill-eval']['status'] = 0
        return base_conf
    base_conf['distill-eval']['args'].update(exp_conf['distill-eval'])
    base_conf['distill-eval']['args']['model_list'] = [{'name': base_conf['distill']['args']['exp_name'], 'path': base_conf['distill']['args']['model_save_dir']}]
    return base_conf

def set_distill_bench_conf(base_conf, exp_conf):
    if 'distill-bench' not in exp_conf:
        base_conf['distill-bench']['status'] = 0
        return base_conf
    base_conf['distill-bench']['args'].update(exp_conf['distill-bench'])
    base_conf['distill-bench']['args']['model'] = base_conf['distill']['args']['model_save_dir']
    return base_conf

def set_distill_conf(base_conf, exp_conf, basic_conf):
    if 'distill' not in exp_conf:
        base_conf['distill']['status'] = 0
        base_conf['distill-eval']['status'] = 0
        base_conf['distill-bench']['status'] = 0
        return base_conf
    base_conf['distill']['args'].update(exp_conf['distill'])
    base_conf['distill']['args']['exp_name'] = basic_conf['experiment']
    base_conf['distill']['args']['teacher_model'] = base_conf['train']['args']['model_save_dir']
    base_conf['distill']['args']['model_save_dir'] = '{}_distill'.format(base_conf['train']['args']['model_save_dir'])
    base_conf = set_distill_eval_conf(base_conf, exp_conf)
    base_conf = set_distill_bench_conf(base_conf, exp_conf)
    return base_conf

def set_export_conf(base_conf, exp_conf, basic_conf):
    # export
    if 'export' not in exp_conf:
        base_conf['export']['status'] = 0
        base_conf['export-bench']['status'] = 0
        base_conf['export-push']['status'] = 0
        return base_conf
    if exp_conf['export']['model_name'] == 'train':
        base_conf['export']['args']['model_name'] = base_conf['train']['args']['model_save_dir']
    else:
        base_conf['export']['args']['model_name'] = base_conf['distill']['args']['model_save_dir']
    base_conf['export']['args']['onnx_path'] = '{}_export'.format(base_conf['export']['args']['model_name'])
    # export-bench
    base_conf['export-bench']['status'] = 1
    base_conf['export-bench']['args'].update(base_conf['export']['args'])
    # export-push
    base_conf['export-push']['status'] = 1
    base_conf['export-push']['args']['model'] = basic_conf['experiment']
    base_conf['export-push']['args']['tag'] = basic_conf['version']
    base_conf['export-push']['args']['onnx_path'] = base_conf['export']['args']['onnx_path']
    return base_conf

def set_conf(base_conf, exp_conf):
    # basic gloabl conf
    basic_conf = set_basic_conf(exp_conf)
    base_conf.update(basic_conf)
    # pipeline conf
    pipe_conf = OrderedDict()
    for step in base_conf['pipeline']:
        pipe_conf[step['name']] = step
    pipe_conf = set_train_conf(pipe_conf, exp_conf, basic_conf)
    pipe_conf = set_distill_conf(pipe_conf, exp_conf, basic_conf)
    pipe_conf = set_export_conf(pipe_conf, exp_conf, basic_conf)
    base_conf['pipeline'] = []
    for name, step in pipe_conf.items():
        base_conf['pipeline'].append(step)
    return base_conf


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-yaml", required=True, help="Experiment config yaml file.")
    parser.add_argument("--pipe-yaml", required=True, help="Pipeline config yaml file.")
    args = parser.parse_args()

    base_yaml = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pipeline.yaml')
    base_conf = yaml.load(open(base_yaml, 'r', encoding='utf8'), Loader=yaml.FullLoader)
    exp_conf = yaml.load(open(args.exp_yaml, 'r', encoding='utf8'), Loader=yaml.FullLoader)

    base_conf = set_conf(base_conf, exp_conf)
    #print(json.dumps(base_conf, indent=4))

    with open(args.pipe_yaml, 'w', encoding='utf8') as fout:
        yaml.dump(base_conf, fout, indent=2, default_flow_style=False, allow_unicode=True, sort_keys=False)
