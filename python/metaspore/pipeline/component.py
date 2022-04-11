import kfp
import os
import string
from ..s3_utils import get_s3_resource, parse_s3_dir_url
from .experiment import *

def to_component_func(
    **kwargs
):
    def decorated_func(func):
        use_user_file = False
        if "s3_path" in kwargs:
            use_user_file = True
            print("upload dependency")
            s3 = get_s3_resource()
            public_bucket, public_dir = parse_s3_dir_url(kwargs["s3_path"]) 
            # dmetasoul-bucket  jinhan/public_dir/
            file_name = os.path.basename(kwargs["algo_path"]) 
            # WideDeep.py
            file_dir = public_dir + file_name 
            # jinhan/public_dir/WideDeep.py

            s3.Bucket(public_bucket).upload_file(kwargs["algo_path"] ,file_dir)

        else:
            print("using exist func in metaspore")

        def component_func(input_string:string):
            if use_user_file == True:
                s3 = get_s3_resource()
                s3.Bucket(public_bucket).download_file(file_dir, '/tmp/'+ file_name)
            
            # parse pipeline parameters
            dic = dict(eval(input_string))
            experiment_dic = dic["experiment"]
            exp_obj_for_runner = Experiment(experiment_dic)
            user_argo_para_dic = dic["user_argo_parameter"]

            func(exp_obj_for_runner,**user_argo_para_dic)

        print("creating ",kwargs["component_name"])
        kfp.components.func_to_container_op(
            func= component_func,
            base_image = 'hub.kce.ksyun.com/dmetasoul/spark:dmetasoul-v1.2.2-test',
            output_component_file= 'kubeflow_components/' + kwargs['component_name'],
            use_code_pickling= True
        )
        return component_func
    return decorated_func