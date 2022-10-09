import os
import statistics
import yaml


class FileUtil:

    @staticmethod
    def write_dict_to_yaml_file(file_path, content):
        FileUtil.check_and_overwrite_file(file_path)
        with open(file_path, "w") as f:
            yaml.dump(content, f, default_flow_style=False)

    @staticmethod
    def write_file(file_path, content):
        FileUtil.check_and_overwrite_file(file_path)
        with open(file_path, "w") as f:
            f.write(content)

    @staticmethod
    def check_and_overwrite_file(file_path):
        if os.path.exists(file_path):
            os.remove(file_path)
        file_dir = os.path.split(file_path)[0]
        if not os.path.isdir(file_dir):
            os.makedirs(file_dir)
