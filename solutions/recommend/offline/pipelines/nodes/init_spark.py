from .node import PipelineNode
import metaspore as ms
import subprocess

class InitSparkNode(PipelineNode):
    def __call__(self, **payload) -> dict:
        conf = payload['conf']
        subprocess.run(['zip', '-r', conf['zip_path'], 'python'], cwd=conf['zip_cwd'])
        spark_confs={
            "spark.network.timeout":"500",
            "spark.ui.showConsoleProgress": "true",
            "spark.kubernetes.executor.deleteOnTermination":"true",
            "spark.submit.pyFiles":"python.zip",
        }
        spark = ms.spark.get_session(local=conf['local'],
                                     app_name=conf['app_name'],
                                     batch_size=conf['batch_size'],
                                     worker_count=conf['worker_count'],
                                     server_count=conf['server_count'],
                                     worker_memory=conf['worker_memory'],
                                     server_memory=conf['server_memory'],
                                     coordinator_memory=conf['coordinator_memory'],
                                     spark_confs=spark_confs)
        sc = spark.sparkContext
        print('Debug -- spark init')
        print('Debug -- version:', sc.version)   
        print('Debug -- applicaitonId:', sc.applicationId)
        print('Debug -- uiWebUrl:', sc.uiWebUrl)
        payload['spark'] = spark
        return payload