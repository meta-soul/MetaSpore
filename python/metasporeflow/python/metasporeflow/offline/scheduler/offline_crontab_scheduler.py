import subprocess

from metasporeflow.offline.scheduler.scheduler import Scheduler
from metasporeflow.offline.utils.file_util import FileUtil


class OfflineCrontabScheduler(Scheduler):
    def __init__(self, schedulers_conf, tasks, local_container_name):
        super().__init__(schedulers_conf, tasks)
        self._local_container_name = local_container_name
        self._local_temp_dir = ".tmp"
        self._docker_temp_dir = "/opt" + "/" + self._local_temp_dir

    def publish(self):
        self._write_local_tmp_dir()

        self._copy_tmp_to_docker_container()

        self._publish_docker_crontab()

        self._exec_docker_crontab_script()

    def _generate_cmd(self):
        # 2022年9月27日 remove --scheduler_time for local model
        # cmd = map(lambda x: x.execute +
        #           " --scheduler_time ${SCHEDULER_TIME}", self._dag_tasks)
        cmd = map(lambda x: x.execute, self._dag_tasks)
        cmd = " \n".join(cmd)
        return cmd

    @property
    def _local_crontab_script_file(self):
        return self._local_temp_dir + "/" + self.name + ".sh"

    @property
    def _docker_crontab_script_file(self):
        return self._docker_temp_dir + "/" + self.name + ".sh"

    def _write_local_tmp_dir(self):
        self._write_crontab_script()

    def _write_crontab_script(self):
        content = self._generate_crontab_script_content()
        FileUtil.write_file(self._local_crontab_script_file, content)

    def _generate_crontab_script_content(self):
        script_header = "#!/bin/bash" + "\n"
        exec_path = "cd /opt/volumes/ecommerce_demo/MetaSpore\n"
        scheduler_time = 'SCHEDULER_TIME="`date --iso-8601=seconds`"' + "\n"
        cmd = self._generate_cmd()
        script_content = script_header + \
            scheduler_time + \
            cmd
        return script_content

    def _copy_tmp_to_docker_container(self):
        src = self._local_temp_dir + "/."
        dst = "%s:%s/" % (self._local_container_name,
                          self._docker_temp_dir)
        overwrite_docker_tmp_dir = "rm -rf %s && mkdir -p %s " % (
            self._docker_temp_dir, self._docker_temp_dir)

        overwrite_docker_tmp_dir_cmd = ['docker', 'exec', '-i', self._local_container_name,
                                        '/bin/bash', '-c', overwrite_docker_tmp_dir]
        copy_tmp_to_docker_cmd = ['docker', 'cp', src, dst]

        subprocess.run(overwrite_docker_tmp_dir_cmd)
        subprocess.run(copy_tmp_to_docker_cmd)

    def _publish_docker_crontab(self):
        crontab_cmd = "\"%s sh %s >> /tmp/%s.log\"" % (self.cronExpr,
                                                       self._docker_crontab_script_file,
                                                       self.name)
        publish_crontab_msg = "crontab -l | { cat; echo %s; } | crontab -" % crontab_cmd
        print("[publish crontab]: \n" +
              "scheduler name: %s \ncrontab_cmd: %s" % (self.name, crontab_cmd))

        publish_docker_crontab_cmd = ['docker', 'exec', '-i', self._local_container_name,
                                      '/bin/bash', '-c', publish_crontab_msg]

        subprocess.run(publish_docker_crontab_cmd)
        # self._get_crontab_list()

    # def _get_crontab_list(self):
    #     get_crontab_list = 'crontab -l'
    #     get_crontab_list_cmd = ['docker', 'exec', '-i', self._local_container_name,
    #                             '/bin/bash', '-c', get_crontab_list]
    #     res = subprocess.run(get_crontab_list_cmd,
    #                          capture_output=True,
    #                          text=True)
    #     msg = "[check crontab list]: \n" + res.stdout
    #     print(msg)

    def _exec_docker_crontab_script(self):
        exec_docker_crontab_script_msg = "sh %s " % (
            self._docker_crontab_script_file)
        msg = "[trigger scheduler once]: \n" + \
            "scheduler name: %s \n" % (self.name,) + \
            "cmd : %s" % (exec_docker_crontab_script_msg)
        print(msg)
        exec_docker_crontab_script_cmd = ['docker', 'exec', '-i', self._local_container_name,
                                          '/bin/bash', '-c', exec_docker_crontab_script_msg]
        subprocess.run(exec_docker_crontab_script_cmd)
