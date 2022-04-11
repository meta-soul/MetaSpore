import imp
import sys
import  metaspore as ms

@ ms.pipeline.component.to_component_func(
    component_name = "WideDeep.yaml"
)
def WideDeep_Experiment(experiment,**kwargs):
    from ..runners.WideDeep import experiment_run_me
    experiment_run_me(experiment,**kwargs)