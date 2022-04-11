import metaspore as ms
import kfp

comp1 = ms.kubeflow.load_component("WideDeep")

pipeline_conf = kfp.dsl.PipelineConf()
pipeline_conf.set_image_pull_policy('Always')

def my_pipeline(parameter):
    comp1(parameter)

kfp.compiler.Compiler().compile(
    pipeline_func = my_pipeline,
    package_path = 'my_pipeline.yaml',
    pipeline_conf=pipeline_conf
)