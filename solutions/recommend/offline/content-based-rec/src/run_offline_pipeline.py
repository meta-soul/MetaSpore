import os
import argparse
import subprocess

import yaml

def create_task(task_name, task_script, task_args, working_dir, stdout=subprocess.PIPE, stderr=subprocess.PIPE):
    cmd = task_script
    for name, arg_value in task_args.items():
        arg_name = name.replace('_', '-')
        arg_name = f'--{arg_name}'
        if arg_value is None:
            cmd.append(arg_name)
        else:
            cmd.extend([arg_name, str(arg_value)])
    task_cmd = ' '.join(map(str, cmd))
    #return task_cmd, None
    env = {
        'PYTHONPATH': working_dir if not os.environ.get('PYTHONPATH', '') else working_dir+":"+os.environ['PYTHONPATH']
    }
    return task_cmd, subprocess.Popen(task_cmd, encoding='utf-8',
        shell=True, stdout=stdout, stderr=stderr, cwd=working_dir, env=env)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, required=True, help='config file path')
    args = parser.parse_args()

    with open(args.conf, 'r') as fin:
        conf = yaml.load(fin, Loader=yaml.FullLoader)

    py_bin = os.path.abspath(conf['python'])
    working_dir = os.path.abspath(conf['working_dir'])
    print("Start pipeline: {}".format(conf['name']))
    print("Working dir: {}".format(working_dir))
    print("Python: {}".format(py_bin))

    for task_conf in conf['pipeline']:
        print("\tTask: {}".format(task_conf['name']))
        print("\t\tRunning: {}".format(task_conf['status']==1))
        if task_conf['status'] != 1:
            continue
        task_cmd, task_proc = create_task(task_conf['name'], [py_bin, task_conf['script']], task_conf['args'], working_dir)
        print(f"\t\tCommand: {task_cmd}")
        #returncode = 0
        returncode = task_proc.wait()
        if returncode == 0:
            print("\t\tResult: success")
        else:
            print("\t\tResult: fail")
            print(f"***pipeline be broken during {task_conf['name']} task!!!")
            break


if __name__ == '__main__':
    main()
