from ..models import WideDeep
import metaspore as ms

def experiment_run_me(experiment,**kwargs):
    module = WideDeep.DemoModule()
    estimator = ms.PyTorchEstimator(module = module)

    experiment.fill_parameter(estimator)

    print(estimator.consul_host)
    def use_s3a(url):
        return url.replace('s3://', 's3a://')

    spark_session = ms.spark.get_session(   batch_size=100,
                                            worker_count=estimator.worker_count,
                                            server_count=estimator.server_count,
                                            spark_confs = 
                                        { 
                                            'spark.hadoop.fs.s3a.endpoint' : 'ks3-cn-beijing-internal.ksyuncs.com',
                                            'spark.hadoop.fs.s3a.signing-algorithm' : 'S3SignerType',
                                            'spark.executorEnv.AWS_REGION' : 'BEIJING',
                                            'spark.kubernetes.driverEnv.AWS_REGION' : 'BEIJING'
                                        },
                                        local=True)

    train_dataset = spark_session.read.parquet(use_s3a('s3://dmetasoul-bucket/demo/' + 'criteo_x1/train_5.parquet'))
    train_dataset = train_dataset.limit(30)

    estimator.fit(train_dataset)