import os
import tarfile
import time

import yaml
from jinja2 import Environment, FileSystemLoader

from .template_treasury import (crontab_scheduler_template, join_data_template,
                                metaspore_flow_template,
                                notify_load_model_template,
                                offline_flow_template, online_flow_template,
                                sync_data_template, train_model_ctr_template,
                                train_model_icf_template,
                                train_model_pop_template)


class SchedulerMode:
   LOCALMODE = "Local"
   K8SCLUSTER = "K8sCluster"
   SAGEMAKERMODE = "SageMaker"


def get_scene_instance(scene_name: str, 
                       scheduler_mode: str = SchedulerMode.LOCALMODE):
    now = int(time.time())
    
    if scheduler_mode == SchedulerMode.SAGEMAKERMODE:
        scheduler_mode = SchedulerMode.SAGEMAKERMODE
        base_dir = ""
    else:
        scheduler_mode = SchedulerMode.LOCALMODE
        base_dir = get_scene_working_dir(scene_name, now)
    
    context = {
        'status': 0,
        'scene_id': scene_name,
        'base_dir': base_dir,
        'scheduler_mode': scheduler_mode,
        'scene_working_dir': get_scene_working_dir(scene_name, now),
        'training_model_types': ['match_popular', 'match_itemcf', 'match_swing', 'rank_deepctr'],
        'training_model_yamls': ['train_model_itemcf', 'train_model_pop', 'train_model_deepctr'],
        'timestamp': now,
        'user_feature_collection': get_scene_mongo_collection_name(scene_name,
            "example", "user_feature", now),
        'item_feature_collection': get_scene_mongo_collection_name(scene_name,
            "example", "item_feature", now),
        'item_summary_collection': get_scene_mongo_collection_name(scene_name,
            "example", "item_summary", now),
        'match_itemcf_dump_collection': get_scene_mongo_collection_name(scene_name,
            "matcher", "match_itemcf", now),
        'match_swing_dump_collection': get_scene_mongo_collection_name(scene_name,
            "matcher", "match_swing", now),
        'match_popular_dump_collection': get_scene_mongo_collection_name(scene_name,
            "matcher", "match_popular", now),
        'source_user_path': get_user_source_path(base_dir, scene_name, scene_name, now),
        'source_item_path': get_item_source_path(base_dir, scene_name, scene_name, now),
        'source_interaction_path': get_interaction_source_path(base_dir, scene_name, scene_name, now),
        'example_match_icf_train_path': get_match_icf_train_path(base_dir, scene_name, scene_name, now),
        'example_match_icf_test_path': get_match_icf_test_path(base_dir, scene_name, scene_name, now),
        'example_ctr_nn_train_path': get_ctr_nn_train_path(base_dir, scene_name, scene_name, now),
        'example_ctr_nn_test_path': get_ctr_nn_test_path(base_dir, scene_name, scene_name, now),
        'example_ctr_gbm_train_path': get_ctr_gbm_train_path(base_dir, scene_name, scene_name, now),
        'example_ctr_gbm_test_path': get_ctr_gbm_test_path(base_dir, scene_name, scene_name, now),
        'example_match_nn_train_path': get_match_nn_train_path(base_dir, scene_name, scene_name, now),
        'example_match_nn_test_path': get_match_nn_test_path(base_dir, scene_name, scene_name, now),
        'example_match_nn_item_path': get_match_nn_item_path(base_dir, scene_name, scene_name, now),
        'rank_deepctr_online_name': get_scene_online_model_name(scene_name, "ranker", "rank_deepctr", now)
    }
    return context


def gen_conf_from_template(out_file, template_file, **kwargs):
    print('generate template')
    print('\tIn :', template_file)
    print('\tOut:', out_file)
    scheduler_dir = os.path.dirname(os.path.abspath(__file__))
    template_dir = os.path.join(scheduler_dir, 'templates')
    templateEnv = Environment(loader=FileSystemLoader(searchpath=template_dir))
    template = templateEnv.get_template(template_file)
    outputText = template.render(**kwargs)
    if out_file:
        with open(out_file, 'w') as text_file:
            text_file.write(outputText)
    return outputText

def load_yaml(file_path):
    with open(file_path, 'r') as stream:
        result = yaml.load(stream, Loader=yaml.FullLoader)
    return result

def load_yaml_str(yaml_str):
    result = yaml.safe_load(yaml_str)
    return result

def save_yaml(data, file_path):
    print('generate template')
    print('\tIn : string')
    print('\tOut:', file_path)
    if file_path:
        dirs = os.path.dirname(file_path)
        if not os.path.exists(dirs):
            os.makedirs(dirs)
        with open(file_path, 'w') as stream:
            yaml.dump(data, stream, default_flow_style=False, allow_unicode=True, sort_keys=False, encoding='utf-8')
    return yaml.dump(data, default_flow_style=False, allow_unicode=True, sort_keys=False, encoding='utf-8')

def get_scene_working_dir(scene_id, now):
    return f"{os.getcwd()}/{scene_id}/{now}"

def get_user_source_path(datadir, scene_id, source_id, now):
    user_path = os.path.join(str(datadir), f"output/data/{scene_id}/{source_id}/{now}/user.parquet")
    return user_path

def get_item_source_path(datadir, scene_id, source_id, now):
    item_path = os.path.join(str(datadir), f"output/data/{scene_id}/{source_id}/{now}/item.parquet")
    return item_path

def get_interaction_source_path(datadir, scene_id, source_id, now):
    iteraction_path = os.path.join(str(datadir), f"output/data/{scene_id}/{source_id}/{now}/interaction.parquet")
    return iteraction_path

def get_ctr_nn_train_path(datadir, scene_id, sample_id, now):
    ctr_nn_train_path = os.path.join(str(datadir), f"output/data/{scene_id}/{sample_id}/{now}/ctr/nn/train.parquet")
    return  ctr_nn_train_path

def get_ctr_nn_test_path(datadir, scene_id, sample_id, now):
    ctr_nn_test_path = os.path.join(str(datadir), f"output/data/{scene_id}/{sample_id}/{now}/ctr/nn/test.parquet")
    return ctr_nn_test_path

def get_match_icf_train_path(datadir, scene_id, sample_id, now):
    match_icf_train_path = os.path.join(str(datadir), f"output/data/{scene_id}/{sample_id}/{now}/match/icf/train.parquet")
    return match_icf_train_path

def get_match_icf_test_path(datadir, scene_id, sample_id, now):
    match_icf_test_path = os.path.join(str(datadir), f"output/data/{scene_id}/{sample_id}/{now}/match/icf/test.parquet")
    return match_icf_test_path

def get_ctr_gbm_train_path(datadir, scene_id, sample_id, now):
    ctr_gbm_train_path = os.path.join(str(datadir), f"output/data/{scene_id}/{sample_id}/{now}/ctr/gbm/train.parquet")
    return  ctr_gbm_train_path

def get_ctr_gbm_test_path(datadir, scene_id, sample_id, now):
    ctr_gbm_test_path = os.path.join(str(datadir), f"output/data/{scene_id}/{sample_id}/{now}/ctr/gbm/test.parquet")
    return ctr_gbm_test_path

def get_match_nn_train_path(datadir, scene_id, sample_id, now):
    match_nn_train_path = os.path.join(str(datadir), f"output/data/{scene_id}/{sample_id}/{now}/match/nn/train.parquet")
    return match_nn_train_path

def get_match_nn_test_path(datadir, scene_id, sample_id, now):
    match_nn_test_path = os.path.join(str(datadir), f"output/data/{scene_id}/{sample_id}/{now}/match/nn/test.parquet")
    return match_nn_test_path
        
def get_match_nn_item_path(datadir, scene_id, source_id, now):
    item_path = os.path.join(str(datadir), f"output/data/{scene_id}/{source_id}/{now}/match/nn/item.parquet")
    return item_path

def get_scene_mongo_collection_name(scene_id, type, name, now):
    return f"scene_{scene_id}_{type}_{name}_{now}"

def get_scene_online_model_name(scene_id, type, name, now):
    return f"scene_{scene_id}_{type}_{name}_{now}"

def copy_codebase_to_dir(working_dir, codebase_tar='volumes.tar.gz'):
    scheduler_dir = os.path.dirname(os.path.abspath(__file__))
    with tarfile.open(os.path.join(scheduler_dir, codebase_tar), 'r:gz') as tfile:
        tfile.extractall(path=working_dir)

def gen_metaspore_flow_conf(scene_context,
                           template_file='./conf_template/metaspore-flow_template.yml',
                           out_file='./output/metaspore-flow.yml'):
    kwargs = {}
    kwargs['working_dir'] = "."

    gen_conf_from_template(out_file, template_file, **kwargs)

    metaspore_flow = load_yaml_str(open(out_file, 'r').read())
    if scene_context["scheduler_mode"] != SchedulerMode.LOCALMODE:
        metaspore_flow["metadata"]["name"] = scene_context["scene_id"]
    metaspore_flow['metadata']['uses'] += list(map(lambda x: f'{kwargs["working_dir"]}/{x}.yml', 
        scene_context['training_model_yamls']))
    metaspore_flow["spec"]["deployMode"] = scene_context["scheduler_mode"]
    if scene_context["scheduler_mode"] == SchedulerMode.LOCALMODE:
        metaspore_flow["spec"]["sharedVolumeInContainer"] = scene_context["scene_working_dir"]
    
    save_yaml(metaspore_flow, out_file)

def gen_sage_maker_conf(template_file='./conf_template/sage_maker_config_template.yaml',
                        out_file='./output/sage_maker_config.yml'):
    gen_conf_from_template(out_file, template_file)

def gen_setup_conf(scene_context,
                   template_file='./conf_template/setup_template.yaml',
                   out_file='./output/setup.yaml'):

    kwargs = {}
    kwargs['user_path'] = scene_context['source_user_path']
    kwargs['item_path'] = scene_context['source_item_path']
    kwargs['interaction_path'] = scene_context['source_interaction_path']

    return gen_conf_from_template(out_file, template_file, **kwargs)

def gen_sample_conf(scene_context,
                    template_file='./conf_template/gen_samples_template.yaml',
                    out_file='./output/gen_samples.yaml'):
    kwargs = {}
    
    kwargs['ctr_nn_train_path'] = get_ctr_nn_train_path(scene_context["base_dir"], scene_context["scene_id"], scene_context["scene_id"], scene_context['timestamp'])
    kwargs['ctr_nn_test_path'] = get_ctr_nn_test_path(scene_context["base_dir"], scene_context["scene_id"], scene_context["scene_id"], scene_context['timestamp'])
    kwargs['ctr_gbm_train_path'] = get_ctr_gbm_train_path(scene_context["base_dir"], scene_context["scene_id"], scene_context["scene_id"], scene_context['timestamp'])
    kwargs['ctr_gbm_test_path'] = get_ctr_gbm_test_path(scene_context["base_dir"], scene_context["scene_id"], scene_context["scene_id"], scene_context['timestamp'])
    kwargs['match_icf_train_path'] = get_match_icf_train_path(scene_context["base_dir"], scene_context["scene_id"], scene_context["scene_id"], scene_context['timestamp'])
    kwargs['match_icf_test_path'] = get_match_icf_test_path(scene_context["base_dir"], scene_context["scene_id"], scene_context["scene_id"], scene_context['timestamp'])
    kwargs['match_nn_train_path'] = get_match_nn_train_path(scene_context["base_dir"], scene_context["scene_id"], scene_context["scene_id"], scene_context['timestamp'])
    kwargs['match_nn_test_path'] = get_match_nn_test_path(scene_context["base_dir"], scene_context["scene_id"], scene_context["scene_id"], scene_context['timestamp'])
    kwargs['match_nn_item_path'] = get_match_nn_item_path(scene_context["base_dir"], scene_context["scene_id"], scene_context["scene_id"], scene_context['timestamp'])
    kwargs['user_feature_collection'] = scene_context['user_feature_collection']
    kwargs['item_feature_collection'] = scene_context['item_feature_collection']
    kwargs['item_summary_collection'] = scene_context['item_summary_collection']
    kwargs['user_path'] = scene_context['source_user_path']
    kwargs['item_path'] = scene_context['source_item_path']
    kwargs['interaction_path'] = scene_context['source_interaction_path']

    return gen_conf_from_template(out_file, template_file, **kwargs)

def gen_model_training_conf(scene_context, template_files, out_files,
        allowed_model_types=['match_popular', 'match_itemcf', 'match_swing', 'rank_deepctr']):
    
    for model_type in allowed_model_types:
        if model_type.startswith('match_'):
            gen_model_training_matcher_conf(scene_context, template_files[model_type], out_files[model_type])
        elif model_type == 'rank_deepctr':
            gen_model_training_deepctr_conf(scene_context, template_files[model_type], out_files[model_type])

def gen_model_training_matcher_conf(scene_context,
                    template_file='./conf_template/swing_i2i_template.yaml',
                    out_file='./output/swing_i2i.yaml',
                    allowed_model_types=['match_popular', 'match_itemcf', 'match_swing']):

    kwargs = {}
    kwargs['train_path'] = scene_context['example_match_icf_train_path']
    kwargs['test_path'] = scene_context['example_match_icf_test_path']

    return gen_conf_from_template(out_file, template_file, **kwargs)  

def gen_model_training_deepctr_conf(scene_context,
                     template_file='./conf_template/deepctr_template.yaml',
                     out_file='./output/deepctr.yaml',
                     allowed_model_types=['rank_deepctr']):
    kwargs = {}
    kwargs['train_path'] = scene_context['example_ctr_nn_train_path']
    kwargs['test_path'] = scene_context['example_ctr_nn_test_path']
    kwargs['base_dir'] = scene_context['base_dir']
    kwargs['rank_deepctr_online_name'] = scene_context['rank_deepctr_online_name']

    return gen_conf_from_template(out_file, template_file, **kwargs)

def gen_online_experiment_conf(ab_conf):
    scenes, layers, experiments = [], [], []
    experiments.append({
        'name': 'experiment.recall.swing', 'then': ['recall_swing']
    })
    experiments.append({
        'name': 'experiment.recall.pop', 'then': ['recall_pop']
    })
    experiments.append({
        'name': 'experiment.related.swing', 'then': ['related_swing']
    })
    # experiments.append({
    #     'name': 'experiment.related.pop', 'then': ['related_pop']
    # })
    experiments.append({
        'name': 'experiment.rank.widedeep', 'then': ['rank_widedeep']
    })
    layers.append({'name': 'layer.recall', 'data': {}})
    for x in ab_conf['guess-you-like']['recall']:
        if x['name'] == 'itemcf':
            layers[-1]['data']['experiment.recall.swing'] = x['weight']
        if x['name'] == 'popular':
            layers[-1]['data']['experiment.recall.pop'] = x['weight']
    layers.append({'name': 'layer.related', 'data': {}})
    for x in ab_conf['looked-and-looked']['recall']:
        if x['name'] == 'itemcf':
            layers[-1]['data']['experiment.related.swing'] = x['weight']
        # if x['name'] == 'popular':
        #     layers[-1]['data']['experiment.related.pop'] = x['weight']
    layers.append({'name': 'layer.rank', 'data': {'experiment.rank.widedeep': 1.0}})
    scenes.append({
        'name': 'guess-you-like', 'layers': ['layer.recall', 'layer.rank'], 'additionalRecalls': ['pop']
    })
    scenes.append({
        'name': 'looked-and-looked', 'layers': ['layer.related', 'layer.rank'], 'additionalRecalls': ['pop']
    })
    return experiments, layers, scenes

def gen_online_flow_conf(scene_context,
                         template_file='./conf_template/online_flow_template.yaml',
                         out_file='./output/online_flow.yml'):
    
    kwargs = {}
    kwargs['user_key'] = "user_id"
    kwargs['item_key'] = "item_id"
    kwargs['user_item_ids'] = "user_bhv_item_seq"
    kwargs['user_item_ids_spilt'] = "\\u0001"
    
    kwargs['rank_model_cross_features'] = list()
    kwargs['user_feature_collection'] = scene_context['user_feature_collection']
    kwargs['item_feature_collection'] = scene_context['item_feature_collection']
    kwargs['item_summary_collection'] = scene_context['item_summary_collection']
    kwargs['match_itemcf_dump_collection'] = scene_context['match_itemcf_dump_collection']
    kwargs['match_swing_dump_collection'] = scene_context['match_swing_dump_collection']
    kwargs['match_popular_dump_collection'] = scene_context['match_popular_dump_collection']
    kwargs['rank_deepctr_online_name'] = scene_context['rank_deepctr_online_name']

    return gen_conf_from_template(out_file, template_file, **kwargs)

def gen_crontab_scheduler(scene_context, save_path):
    crontab_scheduler = load_yaml_str(crontab_scheduler_template)
    if scene_context["scheduler_mode"] == SchedulerMode.SAGEMAKERMODE:
        crontab_scheduler["kind"] = "OfflineSageMakerScheduler"
        crontab_scheduler['spec']['dag']['join_data'] = (scene_context['training_model_yamls'] + 
                                                        crontab_scheduler['spec']['dag']['join_data'])
    elif scene_context["scheduler_mode"] == SchedulerMode.LOCALMODE:
        crontab_scheduler["kind"] = "OfflineCrontabScheduler"
        crontab_scheduler['spec']['dag']['join_data'] = (scene_context['training_model_yamls'] + 
                                                        crontab_scheduler['spec']['dag']['join_data'] + ["notify_load_model"])
    crontab_scheduler['spec']['dag']['sync_data'] = ['join_data']
    save_yaml(crontab_scheduler, save_path)


def gen_metaspore_flow(scene_context, save_path):
   model_type_list = scene_context['training_model_yamls']
   metaspore_flow = load_yaml_str(metaspore_flow_template)
   metaspore_flow['metadata']['uses'] += list(map(lambda x: f'./{x}.yml', model_type_list))
   save_yaml(metaspore_flow, save_path)


def gen_sync_data(save_path, workdir, is_local=False):
    sync_data = load_yaml_str(sync_data_template)
    if is_local:
        script_path = sync_data["spec"]["scriptPath"]
        sync_data["spec"]["scriptPath"] = f"{workdir}/setup/{script_path}"
        config_path = sync_data["spec"]["configPath"]
        sync_data["spec"]["configPath"] = f"{workdir}/setup/conf/{config_path}"
    save_yaml(sync_data, save_path)


def gen_offline_flow(save_path):
    offline_flow = load_yaml_str(offline_flow_template)
    save_yaml(offline_flow, save_path)


def gen_online_flow(save_path):
    online_local_flow = load_yaml_str(online_flow_template)
    save_yaml(online_local_flow, save_path)


def gen_join_data(save_path, workdir, is_local=False):
    join_data = load_yaml_str(join_data_template)
    if is_local:
        script_path = join_data["spec"]["scriptPath"]
        join_data["spec"]["scriptPath"] = f"{workdir}/dataset/{script_path}"
        config_path = join_data["spec"]["configPath"]
        join_data["spec"]["configPath"] = f"{workdir}/dataset/conf/{config_path}"
    save_yaml(join_data, save_path)


def gen_train_model_pop(save_path, workdir, is_local=False):
    train_model = load_yaml_str(train_model_pop_template)
    if is_local:
        script_path = train_model["spec"]["scriptPath"]
        train_model["spec"]["scriptPath"] = f"{workdir}/pop/{script_path}"
        config_path = train_model["spec"]["configPath"]
        train_model["spec"]["configPath"] = f"{workdir}/pop/conf/{config_path}"
    save_yaml(train_model, save_path)


def gen_train_model_icf(save_path, workdir, is_local=False):
    train_model = load_yaml_str(train_model_icf_template)
    if is_local:
        script_path = train_model["spec"]["scriptPath"]
        train_model["spec"]["scriptPath"] = f"{workdir}/itemcf/{script_path}"
        config_path = train_model["spec"]["configPath"]
        train_model["spec"]["configPath"] = f"{workdir}/itemcf/conf/{config_path}"
    save_yaml(train_model, save_path)


def gen_train_model_ctr(save_path, workdir, is_local=False):
    train_model = load_yaml_str(train_model_ctr_template)
    if is_local:
        script_path = train_model["spec"]["scriptPath"]
        train_model["spec"]["scriptPath"] = f"{workdir}/deepctr/{script_path}"
        config_path = train_model["spec"]["configPath"]
        train_model["spec"]["configPath"] = f"{workdir}/deepctr/conf/{config_path}"
    save_yaml(train_model, save_path)


def gen_notify_load_model(save_path, workdir, is_local=False):
    notify_load_model = load_yaml_str(notify_load_model_template)
    if is_local:
        script_path = notify_load_model["spec"]["scriptPath"]
        notify_load_model["spec"]["scriptPath"] = f"{workdir}/{script_path}"
        config_path = notify_load_model["spec"]["configPath"]
        notify_load_model["spec"]["configPath"] = f"{workdir}/deepctr/conf/{config_path}"
    save_yaml(notify_load_model, save_path)
