import os

from .template_engine import (copy_codebase_to_dir, gen_crontab_scheduler,
                              gen_join_data, gen_metaspore_flow_conf,
                              gen_model_training_conf, gen_notify_load_model,
                              gen_offline_flow, gen_online_flow_conf,
                              gen_sage_maker_conf, gen_sample_conf,
                              gen_setup_conf, gen_sync_data,
                              gen_train_model_ctr, gen_train_model_icf,
                              gen_train_model_pop)


def local_operaor(scene_context):
    template_dir = "local"
    base_dir = scene_context['scene_working_dir']
    working_dir = base_dir + '/volumes/ecommerce_demo/MetaSpore/demo/ecommerce'
    copy_codebase_to_dir(base_dir, codebase_tar="./volumes/local_volumes.tar.gz")

    setup_conf_dir = os.path.join(working_dir, 'setup/conf')
    os.makedirs(setup_conf_dir, exist_ok=True)

    sample_conf_dir = os.path.join(working_dir, 'dataset/conf')
    os.makedirs(sample_conf_dir, exist_ok=True)

    training_deepctr_conf_dir = os.path.join(working_dir, 'deepctr/conf')
    os.makedirs(training_deepctr_conf_dir, exist_ok=True)

    training_icf_conf_dir = os.path.join(working_dir, 'itemcf/conf')
    os.makedirs(training_icf_conf_dir, exist_ok=True)

    training_pop_conf_dir = os.path.join(working_dir, 'pop/conf')
    os.makedirs(training_pop_conf_dir, exist_ok=True)
    
    gen_setup_conf(
        scene_context=scene_context,
        template_file=f"./{template_dir}/setup_template.yaml",
        out_file=os.path.join(setup_conf_dir, 'setup.yaml')
    )

    gen_sample_conf(
        scene_context=scene_context,
        template_file=f"./{template_dir}/gen_samples_template.yaml",
        out_file=os.path.join(sample_conf_dir, 'gen_samples.yaml'),
    )

    gen_model_training_conf(scene_context, {
        "match_itemcf": f"./{template_dir}/swing_i2i_template.yaml",
        "match_swing": f"./{template_dir}/swing_i2i_template.yaml",
        "match_popular": f"./{template_dir}/popular_topk_template.yaml",
        "rank_deepctr": f"./{template_dir}/deepctr_template.yaml"
    }, {
        "match_itemcf": os.path.join(training_icf_conf_dir, 'swing_i2i.yaml'),
        "match_swing": os.path.join(training_icf_conf_dir, 'swing_i2i.yaml'),
        "match_popular": os.path.join(training_pop_conf_dir, 'popular_topk.yaml'),
        "rank_deepctr": os.path.join(training_deepctr_conf_dir, "deepctr.yaml")
    })

    gen_metaspore_flow_conf(scene_context,
        template_file=f"./{template_dir}/metaspore-flow_template.yml",
        out_file=os.path.join(base_dir, 'metaspore-flow.yml'))

    gen_online_flow_conf(scene_context, 
        template_file=f"./{template_dir}/online_flow_template.yml",
        out_file=os.path.join(base_dir, 'online_flow.yml'))

    gen_crontab_scheduler(scene_context, os.path.join(base_dir, 'crontab_scheduler.yml'))
    gen_offline_flow(os.path.join(base_dir, 'offline_flow.yml'))
    gen_join_data(os.path.join(base_dir, 'join_data.yml'), working_dir, True)
    gen_train_model_pop(os.path.join(base_dir, 'train_model_pop.yml'), working_dir, True)
    gen_train_model_icf(os.path.join(base_dir, 'train_model_itemcf.yml'), working_dir, True)
    gen_train_model_ctr(os.path.join(base_dir, 'train_model_deepctr.yml'), working_dir, True)
    gen_sync_data(os.path.join(base_dir, 'sync_data.yml'), working_dir, True)
    gen_notify_load_model(os.path.join(base_dir, 'notify_load_model.yml'), working_dir, True)
    

def sagemaker_operaor(scene_context):
    base_dir = scene_context['scene_working_dir']
    working_dir = base_dir
    template_dir = "sagemaker"
    copy_codebase_to_dir(base_dir, codebase_tar="./volumes/sagemaker_volumes.tar.gz")
    
    setup_conf_dir = os.path.join(working_dir, 'volumes')
    sample_conf_dir = os.path.join(working_dir, 'volumes')
    training_deepctr_conf_dir = os.path.join(working_dir, 'volumes')
    training_icf_conf_dir = os.path.join(working_dir, 'volumes')
    training_pop_conf_dir = os.path.join(working_dir, 'volumes')
    gen_setup_conf(
        scene_context=scene_context,
        template_file=f"./{template_dir}/setup_template.yaml",
        out_file=os.path.join(setup_conf_dir, 'setup.yaml')
    )

    gen_sample_conf(
        scene_context=scene_context,
        template_file=f"./{template_dir}/gen_samples_template.yaml",
        out_file=os.path.join(sample_conf_dir, 'gen_samples.yaml'),
    )

    gen_model_training_conf(scene_context, {
        "match_itemcf": f"./{template_dir}/swing_i2i_template.yaml",
        "match_swing": f"./{template_dir}/swing_i2i_template.yaml",
        "match_popular": f"./{template_dir}/popular_topk_template.yaml",
        "rank_deepctr": f"./{template_dir}/deepctr_template.yaml"
    }, {
        "match_itemcf": os.path.join(training_icf_conf_dir, 'swing_i2i.yaml'),
        "match_swing": os.path.join(training_icf_conf_dir, 'swing_i2i.yaml'),
        "match_popular": os.path.join(training_pop_conf_dir, 'popular_topk.yaml'),
        "rank_deepctr": os.path.join(training_deepctr_conf_dir, "deepctr.yaml")
    })

    gen_metaspore_flow_conf(scene_context,
        template_file=f"./{template_dir}/metaspore-flow_template.yml",
        out_file=os.path.join(base_dir, 'metaspore-flow.yml'))

    gen_online_flow_conf(scene_context, 
        template_file=f"./{template_dir}/online_flow_template.yml",
        out_file=os.path.join(base_dir, 'online_flow.yml'))
    
    gen_sage_maker_conf(
        template_file=f"./{template_dir}/sage_maker_config_template.yml",
        out_file=os.path.join(base_dir, 'sage_maker_config.yml'))

    gen_crontab_scheduler(scene_context, os.path.join(base_dir, 'crontab_scheduler.yml'))
    gen_offline_flow(os.path.join(base_dir, 'offline_flow.yml'))
    gen_join_data(os.path.join(base_dir, 'join_data.yml'), working_dir)
    gen_train_model_pop(os.path.join(base_dir, 'train_model_pop.yml'), working_dir)
    gen_train_model_icf(os.path.join(base_dir, 'train_model_itemcf.yml'), working_dir)
    gen_train_model_ctr(os.path.join(base_dir, 'train_model_deepctr.yml'), working_dir)
    gen_sync_data(os.path.join(base_dir, 'sync_data.yml'), working_dir)
    # gen_notify_load_model(os.path.join(base_dir, 'notify_load_model.yml'))