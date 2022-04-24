import os
import sys
import logging
import subprocess

import yaml

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def get_abspath(path):
    return os.path.abspath(os.path.expanduser(os.path.expandvars(path)))

def yaml2cmd(task_name, name, values, input_dir, output_dir):
    if task_name in ['train', 'distill']:
        if name in ['train_file', 'eval_file', 'dev_file', 'test_file']:
            value = os.path.abspath(os.path.join(input_dir, values))
        elif name in ['model_save_dir']:
            value = os.path.abspath(os.path.join(output_dir, values))
        elif name in ['teacher_model', 'student_model']:
            value = os.path.abspath(os.path.join(output_dir, values)) if values else ''
        else:
            value = values
    elif task_name in ['train-eval', 'distill-eval']:
        if name == 'eval_list':
            value = ','.join(['{}#{}'.format(x['name'], os.path.abspath(os.path.join(input_dir, x['path']))) for x in values])
        elif name == 'model_list':
            value = ','.join(['{}#{}'.format(x['name'], os.path.abspath(os.path.join(output_dir, x['path']))) for x in values])
        else:
            value = values
    elif task_name in ['train-bench', 'distill-bench']:
        if name == 'model':
            value = os.path.abspath(os.path.join(output_dir, values))
        elif name == 'input_file':
            value = os.path.abspath(os.path.join(input_dir, values))
        else:
            value = values
    elif task_name in ['export', 'export-bench', 'export-push']:
        if name in ['model_name', 'onnx_path']:
            value = os.path.abspath(os.path.join(output_dir, values))
        else:
            value = values
    else:
        value = values

    name = name.replace('_', '-')
    name = f'--{name}'
    return name, value

def create_task(task_name, task_script, task_args, working_dir, input_dir, output_dir, stdout=subprocess.PIPE, stderr=subprocess.PIPE):
    cmd = task_script
    for name, values in task_args.items():
        arg_name, arg_value = yaml2cmd(task_name, name, values, input_dir, output_dir)
        if arg_value is None:
            cmd.append(arg_name)
        else:
            cmd.extend([arg_name, str(arg_value)])
    task_cmd = ' '.join(map(str, cmd))
    return task_cmd, subprocess.Popen(task_cmd, encoding='utf-8', 
        shell=True, stdout=stdout, stderr=stderr, cwd=working_dir, env={})


if __name__ == '__main__':
    yaml_file = sys.argv[1]
    conf = yaml.load(open(yaml_file, 'r', encoding='utf8'), Loader=yaml.FullLoader)
    experiment = conf['experiment']
    working_dir = get_abspath(conf['working_dir'])
    input_dir = get_abspath(conf['input_dir'])
    output_dir = get_abspath(conf['output_dir'])
    log_dir = get_abspath(conf['log_dir'])
    py_bin = get_abspath(conf['python'])
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    logger.info("Expriment: {}".format(experiment))
    for task_conf in conf['pipeline']:
        logger.info("\tTask: {}".format(task_conf['name']))
        logger.info("\t\tRunning: {}".format(task_conf['status']==1))
        if task_conf['status'] != 1:
            continue
        out_path = os.path.join(log_dir, '{}.out'.format(task_conf['name']))
        err_path = os.path.join(log_dir, '{}.err'.format(task_conf['name']))
        out_file = open(out_path, 'w', encoding='utf8')
        err_file = open(err_path, 'w', encoding='utf8')
        logger.info(f"\t\tStdout: {out_path}")
        logger.info(f"\t\tStderr: {err_path}")
        task_cmd, task_proc = create_task(task_conf['name'], [py_bin, task_conf['script']], task_conf['args'],
            working_dir, input_dir, output_dir, stdout=out_file, stderr=err_file)
        logger.info(f"\t\tCommand: {task_cmd}")
        returncode = task_proc.wait()
        out_file.close()
        err_file.close()
        if returncode == 0:
            logger.info("\t\tResult: success")
        else:
            logger.info("\t\tResult: fail")
            logger.warning("***experiment broken during {} task!!! please check log file: {}".format(task_conf['name'], err_path))
            break
