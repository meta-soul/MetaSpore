import imp
import sys
import  metaspore as ms

@ ms.pipeline.component.to_component_func(
    component_name = "WideDeep"
)
def WideDeep_Experiment(experiment,**kwargs):
    import metaspore.algos.runners.WideDeep as WideDeep
    WideDeep.experiment_run_me(experiment,**kwargs)