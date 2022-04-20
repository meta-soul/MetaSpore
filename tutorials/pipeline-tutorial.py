import metaspore as ms
import kfp

comp1 = ms.kubeflow.load_component(
    "WideDeep",
    use_builin_component= True)

pipeline_conf = kfp.dsl.PipelineConf()
pipeline_conf.set_image_pull_policy('Always')

def my_pipeline(parameter):
    task_never_use_cache = comp1(parameter)
    task_never_use_cache.execution_options.caching_strategy.max_cache_staleness = "P0D"

kfp.compiler.Compiler().compile(
    pipeline_func = my_pipeline,
    package_path = 'my_pipeline.yaml',
    pipeline_conf=pipeline_conf
)