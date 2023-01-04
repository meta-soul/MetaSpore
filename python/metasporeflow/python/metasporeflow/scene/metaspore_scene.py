#
# Copyright 2022 DMetaSoul
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#


import os
import tarfile
from typing import List

import yaml
from metasporeflow.scene.models import *


class GenerateFlow:
    def __init__(self, scene_name: str, 
                 scheduler_mode: str=SchedulerMode.LOCALMODE) -> None: 
        self.scene_name = scene_name
        self._metaspore = MetaSporeFlow(
            metadata=MetaSporeFlowMetadata(name=scene_name),
            spec=MetaSporeFlowSpec(deployMode=scheduler_mode)
        )
        self._export_lists={
            "metaspore-flow": self._metaspore
        }
    
    def set_config(self, config: SageMakerConfig) -> None:   
        _sagemakerconfig = MetaSporeFlow(
            kind=MetaSporeFlowKind.SAGEMAKERCONFIG,
            metadata=MetaSporeFlowMetadata(name=f"{self.scene_name}_config"),
            spec=config
        )
        self._metaspore.metadata.uses.append("./sage_maker_config.yml")
        self._export_lists["sage_maker_config"] = _sagemakerconfig
        
    def set_offline_training(self, offline_training: OfflineTraining) -> None:
        self._offline_training = MetaSporeFlow(
            kind=MetaSporeFlowKind.OFFILINETRAINING,
            metadata=MetaSporeFlowMetadata(name=f"{self.scene_name}_offline_training"),
            spec=offline_training
        )
        self._metaspore.metadata.uses.append("./offline/offline_training.yml")
        self._export_lists["offline/offline_training"] = self._offline_training
    
    def add_offline_training_dag(self, sync_data: List[MetaSporeTask] = None, 
                                 join_data: List[MetaSporeTask] = None) -> None:
        for task in sync_data:
            sync_data_task = MetaSporeFlow(
                        kind=MetaSporeFlowKind.OFFILINEPYTHONTASK,
                        metadata=MetaSporeFlowMetadata(name=f"{task.name}_task"),
                        spec=OfflineTrainingPythonTask(configPath=f"{task.name}.yml", scriptPath=task.scriptPath), 
                    )
            self._export_lists[f"offline/task/{task.name}"] = sync_data_task
            self._offline_training.metadata.uses.append(f"./task/{task.name}.yml")
            
            res = MetaSporeFlow(
                    kind=MetaSporeFlowKind.TRAININGJOB,
                    metadata=MetaSporeFlowMetadata(name=f"{task.name}_res", category=task.category),
                    spec=task.resource, 
                )
            self._export_lists[f"volumes/{task.name}"] = res

        for task in join_data:
            join_data_task = MetaSporeFlow(
                        kind=MetaSporeFlowKind.OFFILINEPYTHONTASK,
                        metadata=MetaSporeFlowMetadata(name=f"{task.name}_task"),
                        spec=OfflineTrainingPythonTask(configPath=f"{task.name}.yml", scriptPath=task.scriptPath),
                    )
            
            self._export_lists[f"offline/task/{task.name}"] = join_data_task
            self._offline_training.metadata.uses.append(f"./task/{task.name}.yml")
            
            res = MetaSporeFlow(
                    kind=MetaSporeFlowKind.TRAININGJOB,
                    metadata=MetaSporeFlowMetadata(name=f"{task.name}_res", category=task.category),
                    spec=task.resource, 
                )
            self._export_lists[f"volumes/{task.name}"] = res
        
        self._offline_training.spec.dag[f"{sync_data[0].name}_task"] = [f"{task.name}_task" for task in sync_data[1:]]
        self._offline_training.spec.dag[f"{sync_data[-1].name}_task"] = [f"{task.name}_task" for task in join_data]
    
    def set_online_predict(self, online_predict: OnlinePredict) -> None:
        _online_predict = MetaSporeFlow(
            kind=MetaSporeFlowKind.ONLINEPREDICT,
            metadata=MetaSporeFlowMetadata(name=f"{self.scene_name}_online_predict"),
            spec=online_predict
        )
        self._metaspore.metadata.uses.append("./online/online_predict.yml")
        self._export_lists["online/online_predict"] = _online_predict
    
    def to_yaml(self, saved_path: str="."):
        _extract_code(f"{saved_path}/{self.scene_name}", 
                     codebase_tar="./volumes/sagemaker_volumes.tar.gz")
        for k, v in self._export_lists.items():
            if not os.path.exists(os.path.dirname(f"{saved_path}/{self.scene_name}/{k}.yml")):
                os.makedirs(os.path.dirname(f"{saved_path}/{self.scene_name}/{k}.yml"))
            with open(f"{saved_path}/{self.scene_name}/{k}.yml", "w") as f:
                yaml.dump(v.dict(), f, sort_keys=False)


def _extract_code(working_dir, codebase_tar='volumes.tar.gz'):
    scheduler_dir = os.path.dirname(os.path.abspath(__file__))
    with tarfile.open(os.path.join(scheduler_dir, codebase_tar), 'r:gz') as tfile:
        tfile.extractall(path=working_dir)

def init_flow(name: str, scheduler_mode: SchedulerMode):
    scene = Scene(name=name)
    
    if scheduler_mode == SchedulerMode.SAGEMAKERMODE:
        scene.scheduler_mode = SceneScedulerModel(name=scheduler_mode,
                                                  config=SageMakerConfig())
        
    with open(f"./{name}.yml", "w") as f:
        yaml.dump(scene.dict(), f, sort_keys=False)

def _gen_schema(working_dir: str, schema_fields: list):
    with open(working_dir, "w") as f:
        f.write("\n".join(schema_fields))

def generate_flow(values_file: str):
    scene = None
    with open(values_file) as f:
        scene = Scene.parse_obj(yaml.safe_load(f))
    
    algo = GenerateFlow(scene.name, scene.scheduler_mode.name)
    if  scene.scheduler_mode.name == SchedulerMode.SAGEMAKERMODE:
        algo.set_config(SageMakerConfig.parse_obj(scene.scheduler_mode.config))
    algo.set_offline_training(OfflineTraining(
        cronExpr="* */6 * * *",
        dag={}
    ))
    
    user_table = dict(filter(lambda v : v[1] not in ["", '', None], 
                             scene.datasource.user_tab_info.dict().items()))
    item_table = dict(filter(lambda v : v[1] not in ["", '', None], 
                             scene.datasource.item_tab_info.dict().items()))
    interaction_table = scene.datasource.trans_tab_info.dict()
    
    item_features = summary_features = ["item_id", "brand", "category"]
    for key in item_table.keys():
        if key == "title_column_name":
            summary_features.append("title")
        if key == "url_column_name":
            summary_features.append("url")
        if key == "price_column_name":
            summary_features.append("price")
        if key == "description_column_name":
            summary_features.append("description")
    
    algo.add_offline_training_dag(
        sync_data=[
            MetaSporeTask(name="setup", scriptPath="setup.py", category="Setup",
                          resource=TrainingJobResource(
                              spark=TrainingJobSparkResource(
                                  session_confs=TrainingJobSparkSessionResource(
                                      app_name="Ecommerce Setup",
                                      worker_count=8,
                                      worker_cpu=1,
                                      server_count=8,
                                      server_cpu=1,
                                      coordinator_memory="16G"), 
                                  extended_confs={
                                      "spark.network.timeout": '500',
                                      "spark.ui.showConsoleProgress": 'true',
                                      "spark.hadoop.fs.s3a.aws.credentials.provider": "com.amazonaws.auth.DefaultAWSCredentialsProviderChain",
                                      "spark.default.parallelism": '16',
                                      "spark.sql.shuffle.partitions": '16',
                                      "spark.hadoop.fs.s3a.committer.name": 'directory',
                                      "spark.hadoop.fs.s3a.committer.threads": '16',
                                      "spark.hadoop.mapreduce.fileoutputcommitter.algorithm.version": '2',
                                      "spark.hadoop.fs.s3a.committer.staging.conflict-mode": 'append',
                                      "spark.hadoop.fs.s3a.threads.max": '32',
                                      "spark.hadoop.fs.s3a.connection.maximum": '32',
                                      "spark.hadoop.mapreduce.outputcommitter.factory.scheme.s3a": 'org.apache.hadoop.fs.s3a.commit.S3ACommitterFactory',
                                        }), 
                              load_dataset=TrainingJobLoadDataset(
                                  host=scene.datasource.connect_info.host, 
                                  port=scene.datasource.connect_info.port, 
                                  database=scene.datasource.connect_info.database, 
                                  user=scene.datasource.connect_info.user, 
                                  password=scene.datasource.connect_info.password,
                                  user_table=user_table, 
                                  item_table=item_table,
                                  interaction_table=interaction_table),
                              save_dataset=TrainingJobSaveDataset(
                                  user_path="@{METASPORE_FLOW_S3_WORK_DIR}/flow/scene/@{METASPORE_FLOW_SCENE_NAME}/data/raw/@{METASPORE_FLOW_MODEL_VERSION}/amazon_fashion_user.parquet",
                                  item_path="@{METASPORE_FLOW_S3_WORK_DIR}/flow/scene/@{METASPORE_FLOW_SCENE_NAME}/data/raw/@{METASPORE_FLOW_MODEL_VERSION}/amazon_fashion_item.parquet",
                                  interaction_path="@{METASPORE_FLOW_S3_WORK_DIR}/flow/scene/@{METASPORE_FLOW_SCENE_NAME}/data/raw/@{METASPORE_FLOW_MODEL_VERSION}/amazon_fashion_interaction.parquet"))),
            MetaSporeTask(name="gen_samples", scriptPath="gen_samples.py", category="Dataset/SampleGeneration",
                          resource=GenSampleResource(
                              spark=TrainingJobSparkResource(
                                  session_confs=TrainingJobSparkSessionResource(
                                      app_name="Ecommerce Generating Samples",
                                      worker_count=8,
                                      worker_cpu=1,
                                      server_count=8,
                                      server_cpu=1,
                                      coordinator_memory="16G"),
                                  extended_confs={
                                        "spark.network.timeout": '500',
                                        "spark.ui.showConsoleProgress": 'true',
                                        "spark.hadoop.fs.s3a.aws.credentials.provider": "com.amazonaws.auth.DefaultAWSCredentialsProviderChain",
                                        "spark.default.parallelism": '16',
                                        "spark.sql.shuffle.partitions": '16',
                                        "spark.hadoop.fs.s3a.committer.name": 'directory',
                                        "spark.hadoop.fs.s3a.committer.threads": '16',
                                        "spark.hadoop.mapreduce.fileoutputcommitter.algorithm.version": '2',
                                        "spark.hadoop.fs.s3a.committer.staging.conflict-mode": 'append',
                                        "spark.hadoop.fs.s3a.threads.max": '32',
                                        "spark.hadoop.fs.s3a.connection.maximum": '32',
                                        "spark.hadoop.mapreduce.outputcommitter.factory.scheme.s3a": 'org.apache.hadoop.fs.s3a.commit.S3ACommitterFactory',
                                        }),
                              load_dataset=TrainingJobSaveDataset(
                                  user_path="@{METASPORE_FLOW_S3_WORK_DIR}/flow/scene/@{METASPORE_FLOW_SCENE_NAME}/data/raw/@{METASPORE_FLOW_MODEL_VERSION}/amazon_fashion_user.parquet",
                                  item_path="@{METASPORE_FLOW_S3_WORK_DIR}/flow/scene/@{METASPORE_FLOW_SCENE_NAME}/data/raw/@{METASPORE_FLOW_MODEL_VERSION}/amazon_fashion_item.parquet",
                                  interaction_path="@{METASPORE_FLOW_S3_WORK_DIR}/flow/scene/@{METASPORE_FLOW_SCENE_NAME}/data/raw/@{METASPORE_FLOW_MODEL_VERSION}/amazon_fashion_interaction.parquet"),
                              join_dataset=TrainingJobJoinDataset(
                                  join_on={"user_key": "user_id", "item_key": "item_id", "timestamp": "timestamp"},
                                  user_bhv_seq={"max_len": 10},
                                  negative_sample={"sample_ratio": 3}),
                              gen_feature={"reserve_only_cate_cols": True},
                              gen_sample=[
                                  TrainingJobSample(model_type="ctr_nn", split_test=0.15, shuffle=True, fmt="parquet",
                                                    train_path="@{METASPORE_FLOW_S3_WORK_DIR}/flow/scene/@{METASPORE_FLOW_SCENE_NAME}/data/ctr/nn/@{METASPORE_FLOW_MODEL_VERSION}/train.parquet",
                                                    test_path="@{METASPORE_FLOW_S3_WORK_DIR}/flow/scene/@{METASPORE_FLOW_SCENE_NAME}/data/ctr/nn/@{METASPORE_FLOW_MODEL_VERSION}/test.parquet"),
                                  TrainingJobSample(model_type="ctr_gbm", split_test=0.15, shuffle=True, fmt="parquet",
                                                    train_path="@{METASPORE_FLOW_S3_WORK_DIR}/flow/scene/@{METASPORE_FLOW_SCENE_NAME}/data/ctr/gbm/@{METASPORE_FLOW_MODEL_VERSION}/train.parquet",
                                                    test_path="@{METASPORE_FLOW_S3_WORK_DIR}/flow/scene/@{METASPORE_FLOW_SCENE_NAME}/data/ctr/gbm/@{METASPORE_FLOW_MODEL_VERSION}/test.parquet",
                                                    combine_schema={"user_cols": ["user_id"], "item_cols": ["item_id", "category"],"combine_cols": []}),
                                  TrainingJobSample(model_type="match_nn", split_test=0.15, shuffle=True, fmt="parquet",
                                                    train_path="@{METASPORE_FLOW_S3_WORK_DIR}/flow/scene/@{METASPORE_FLOW_SCENE_NAME}/data/match/nn/@{METASPORE_FLOW_MODEL_VERSION}/train.parquet",
                                                    test_path="@{METASPORE_FLOW_S3_WORK_DIR}/flow/scene/@{METASPORE_FLOW_SCENE_NAME}/data/match/nn/@{METASPORE_FLOW_MODEL_VERSION}/test.parquet",
                                                    item_path="@{METASPORE_FLOW_S3_WORK_DIR}/flow/scene/@{METASPORE_FLOW_SCENE_NAME}/data/match/nn/@{METASPORE_FLOW_MODEL_VERSION}/item.parquet"),
                                  TrainingJobSample(model_type="match_icf", split_test=0.15, shuffle=True, fmt="parquet",
                                                    train_path="@{METASPORE_FLOW_S3_WORK_DIR}/flow/scene/@{METASPORE_FLOW_SCENE_NAME}/data/match/icf/@{METASPORE_FLOW_MODEL_VERSION}/train.parquet",
                                                    test_path="@{METASPORE_FLOW_S3_WORK_DIR}/flow/scene/@{METASPORE_FLOW_SCENE_NAME}/data/match/icf/@{METASPORE_FLOW_MODEL_VERSION}/test.parquet"),
                                  ],
                              dump_nn_feature=TrainingJobFeature(
                                  mongodb=TrainingJobMongo(uri=f"mongodb://{scene.middle_result_storage.user}:{scene.middle_result_storage.password}@{scene.middle_result_storage.host}:{scene.middle_result_storage.port}/?authSource={scene.middle_result_storage.authsource}", database=scene.middle_result_storage.database),
                                  tables=[TrainingJobFeatureTable(feature_column=["user_id", "user_bhv_item_seq"],
                                                                  mongo_collection=f"{scene.name}_user_feature",
                                                                  index_fields=["user_id"], drop_duplicates_by=["user_id"]),
                                          TrainingJobFeatureTable(feature_column=item_features,
                                                                  mongo_collection=f"{scene.name}_item_feature",
                                                                  index_fields=["item_id"], drop_duplicates_by=["item_id"]),
                                          TrainingJobFeatureTable(feature_column=summary_features,
                                                                  mongo_collection=f"{scene.name}_item_summary",
                                                                  index_fields=["item_id"], drop_duplicates_by=["item_id"])
                                          ]
                                  ),
                              dump_lgb_feaure=TrainingJobFeature(
                                  mongodb=TrainingJobMongo(uri=f"mongodb://{scene.middle_result_storage.user}:{scene.middle_result_storage.password}@{scene.middle_result_storage.host}:{scene.middle_result_storage.port}/?authSource={scene.middle_result_storage.authsource}",
                                                           database=scene.middle_result_storage.database),
                                  tables=[TrainingJobFeatureTable(feature_column="user_cols", 
                                                                  mongo_collection=f"{scene.name}_user_lgb_feature", 
                                                                  index_fields=["user_id"]),
                                          TrainingJobFeatureTable(feature_column="item_cols", 
                                                                  mongo_collection=f"{scene.name}_item_lgb_feature",
                                                                  index_fields=["item_id"])]
                                  )))],
        join_data=[
            MetaSporeTask(name="popular_topk", scriptPath="popular_topk.py", category="Match/Popular",
                          resource=TrainingJobResource(
                              spark=TrainingJobSparkResource(
                                  session_confs=TrainingJobSparkSessionResource(
                                      app_name="Ecommerce Popular Pipeline Demo",
                                      worker_count=8,
                                      worker_cpu=1,
                                      server_count=8,
                                      server_cpu=1,
                                      coordinator_memory="16G"),
                                  extended_confs={
                                        "spark.network.timeout": '500',
                                        "spark.ui.showConsoleProgress": 'true',
                                        "spark.hadoop.fs.s3a.aws.credentials.provider": "com.amazonaws.auth.DefaultAWSCredentialsProviderChain",
                                        "spark.default.parallelism": '16',
                                        "spark.sql.shuffle.partitions": '16',
                                        "spark.hadoop.fs.s3a.committer.name": 'directory',
                                        "spark.hadoop.fs.s3a.committer.threads": '16',
                                        "spark.hadoop.mapreduce.fileoutputcommitter.algorithm.version": '2',
                                        "spark.hadoop.fs.s3a.committer.staging.conflict-mode": 'append',
                                        "spark.hadoop.fs.s3a.threads.max": '32',
                                        "spark.hadoop.fs.s3a.connection.maximum": '32',
                                        "spark.hadoop.mapreduce.outputcommitter.factory.scheme.s3a": 'org.apache.hadoop.fs.s3a.commit.S3ACommitterFactory',
                                        }),
                              dataset=TrainingJobDataset(
                                  train_path="@{METASPORE_FLOW_S3_WORK_DIR}/flow/scene/@{METASPORE_FLOW_SCENE_NAME}/data/match/icf/@{METASPORE_FLOW_MODEL_VERSION}/train.parquet"),
                              training=TrainingJobTraining(
                              estimator_params={"max_recommendation_count": 20, "group_nums": 10}),
                              mongodb=TrainingJobMongo(
                                  uri=f"mongodb://{scene.middle_result_storage.user}:{scene.middle_result_storage.password}@{scene.middle_result_storage.host}:{scene.middle_result_storage.port}/?authSource={scene.middle_result_storage.authsource}",
                                  database=scene.middle_result_storage.database, collection=f"{scene.name}_pop", index_fields=["key"]))),
            MetaSporeTask(name="itemcf", scriptPath="itemcf.py", category="Match/I2I/Swing",
                          resource=TrainingJobResource(logging=TrainingJobLogging(loglevel="debug"),
                                                       spark=TrainingJobSparkResource(
                                                            session_confs=TrainingJobSparkSessionResource(
                                                                app_name="Ecommerce Swing I2I Pipeline",
                                                                worker_count=8,
                                                                worker_cpu=1,
                                                                server_count=8,
                                                                server_cpu=1,
                                                                coordinator_memory="16G"),
                                                            extended_confs={
                                                                    "spark.network.timeout": '500',
                                                                    "spark.ui.showConsoleProgress": 'true',
                                                                    "spark.hadoop.fs.s3a.aws.credentials.provider": "com.amazonaws.auth.DefaultAWSCredentialsProviderChain",
                                                                    "spark.default.parallelism": '16',
                                                                    "spark.sql.shuffle.partitions": '16',
                                                                    "spark.hadoop.fs.s3a.committer.name": 'directory',
                                                                    "spark.hadoop.fs.s3a.committer.threads": '16',
                                                                    "spark.hadoop.mapreduce.fileoutputcommitter.algorithm.version": '2',
                                                                    "spark.hadoop.fs.s3a.committer.staging.conflict-mode": 'append',
                                                                    "spark.hadoop.fs.s3a.threads.max": '32',
                                                                    "spark.hadoop.fs.s3a.connection.maximum": '32',
                                                                    "spark.hadoop.mapreduce.outputcommitter.factory.scheme.s3a": 'org.apache.hadoop.fs.s3a.commit.S3ACommitterFactory',
                                                                    }),
                                                       dataset=TrainingJobDataset(train_path="@{METASPORE_FLOW_S3_WORK_DIR}/flow/scene/@{METASPORE_FLOW_SCENE_NAME}/data/ctr/nn/@{METASPORE_FLOW_MODEL_VERSION}/train.parquet", test_path="@{METASPORE_FLOW_S3_WORK_DIR}/flow/scene/@{METASPORE_FLOW_SCENE_NAME}/data/ctr/nn/@{METASPORE_FLOW_MODEL_VERSION}/test.parquet"),
                                                       training=ItemCFModel(
                                                           i2i_estimator_class="metaspore.algos.item_cf_retrieval.ItemCFEstimator",
                                                           i2i_estimator_config_class="metaspore.algos.pipeline.ItemCFEstimatorConfig",
                                                           estimator_params={"max_recommendation_count": 20}),
                                                       mongodb=TrainingJobMongo(
                                                           uri=f"mongodb://{scene.middle_result_storage.user}:{scene.middle_result_storage.password}@{scene.middle_result_storage.host}:{scene.middle_result_storage.port}/?authSource={scene.middle_result_storage.authsource}",
                                                           database=scene.middle_result_storage.database, collection=f"{scene.name}_swing", index_fields=["key"]))),
            MetaSporeTask(name="deepctr", scriptPath="deepctr.py", category="Rank/DeepCTR/WideDeep", 
                          resource=TrainingJobResource(
                              spark=TrainingJobSparkResource(
                                  session_confs=TrainingJobSparkSessionResource(
                                      app_name="Ecommerce Deep CTR Pipeline",
                                      worker_count=8,
                                      worker_cpu=1,
                                      server_count=8,
                                      server_cpu=1,
                                      coordinator_memory="16G"),
                                  extended_confs={
                                        "spark.network.timeout": '500',
                                        "spark.ui.showConsoleProgress": 'true',
                                        "spark.hadoop.fs.s3a.aws.credentials.provider": "com.amazonaws.auth.DefaultAWSCredentialsProviderChain",
                                        "spark.default.parallelism": '16',
                                        "spark.sql.shuffle.partitions": '16',
                                        "spark.hadoop.fs.s3a.committer.name": 'directory',
                                        "spark.hadoop.fs.s3a.committer.threads": '16',
                                        "spark.hadoop.mapreduce.fileoutputcommitter.algorithm.version": '2',
                                        "spark.hadoop.fs.s3a.committer.staging.conflict-mode": 'append',
                                        "spark.hadoop.fs.s3a.threads.max": '32',
                                        "spark.hadoop.fs.s3a.connection.maximum": '32',
                                        "spark.hadoop.mapreduce.outputcommitter.factory.scheme.s3a": 'org.apache.hadoop.fs.s3a.commit.S3ACommitterFactory',
                                        }),
                              dataset=TrainingJobDataset(
                                  train_path="@{METASPORE_FLOW_S3_WORK_DIR}/flow/scene/@{METASPORE_FLOW_SCENE_NAME}/data/ctr/nn/@{METASPORE_FLOW_MODEL_VERSION}/train.parquet",
                                  test_path="@{METASPORE_FLOW_S3_WORK_DIR}/flow/scene/@{METASPORE_FLOW_SCENE_NAME}/data/ctr/nn/@{METASPORE_FLOW_MODEL_VERSION}/test.parquet"),
                              training=DeepCTRModel(
                              deep_ctr_model_class="metaspore.algos.widedeep_net.WideDeep", 
                              estimator_config_class="metaspore.algos.pipeline.DeepCTREstimatorConfig",
                              model_config_class="metaspore.algos.pipeline.WideDeepConfig",
                              model_params={
                                    "wide_combine_schema_path": "./deepctr_wide_combine_schema.txt",
                                    "deep_combine_schema_path": "./deepctr_deep_combine_schema.txt",
                                    "use_wide": True,
                                    "use_dnn": True,
                                    "wide_embedding_dim": 16,
                                    "deep_embedding_dim": 16,
                                    "ftrl_l1": 1.0,
                                    "ftrl_l2": 120.0,
                                    "ftrl_alpha": 1.0,
                                    "ftrl_beta": 1.0,
                                    "dnn_hidden_units": [256, 256, 256],
                                    "sparse_init_var": 0.01,
                                    "dnn_hidden_activations": "ReLU",
                                    "use_bias": True,
                                    "batch_norm": True,
                                    "net_dropout": 0,
                                    "net_regularizer": None
                              },
                              estimator_params={
                                    "model_in_path": "@{METASPORE_FLOW_S3_WORK_DIR}/flow/scene/@{METASPORE_FLOW_SCENE_NAME}/model/checkpoint/@{METASPORE_FLOW_LAST_MODEL_VERSION}/@{METASPORE_FLOW_MODEL_NAME}/",
                                    "model_out_path": "@{METASPORE_FLOW_S3_WORK_DIR}/flow/scene/@{METASPORE_FLOW_SCENE_NAME}/model/checkpoint/@{METASPORE_FLOW_MODEL_VERSION}/@{METASPORE_FLOW_MODEL_NAME}/",
                                    "model_export_path": "@{METASPORE_FLOW_S3_WORK_DIR}/flow/scene/@{METASPORE_FLOW_SCENE_NAME}/model/export/@{METASPORE_FLOW_MODEL_VERSION}/@{METASPORE_FLOW_MODEL_NAME}/",
                                    "model_version": "@{METASPORE_FLOW_MODEL_VERSION}",
                                    "experiment_name": "@{METASPORE_FLOW_MODEL_NAME}",
                                    "input_label_column_index": 0,
                                    "metric_update_interval": 100,
                                    "adam_learning_rate": 0.0001,
                                    "training_epoches": 1,
                                    "shuffle_training_dataset": True
                              })))
        ])

    online_predict = OnlinePredict()
    if  scene.scheduler_mode.name == SchedulerMode.LOCALMODE:
        online_predict.dockers = {
            "recommend": OnlinePredictDocker(image="swr.cn-southwest-2.myhuaweicloud.com/dmetasoul-repo/recommend-service:1.0.0",
                                            ports=[ 13013 ], 
                                            options={"mongo_service": "mongodb.saas-demo", "mongo_port": 27017, "domain": "huawei.dmetasoul.com", "namespace": "saas-demo"}),
            "model": OnlinePredictDocker(image="swr.cn-southwest-2.myhuaweicloud.com/dmetasoul-repo/metaspore-serving-release:cpu-v1.0.1", 
                                        ports=[ 50000 ], 
                                        options={"watch_image": "swr.cn-southwest-2.myhuaweicloud.com/dmetasoul-repo/consul-watch-load:v1.0.0", "watch_port": 8080, "consul_key": "dev/", "endpoint_url": "http://obs.cn-southwest-2.myhuaweicloud.com", "docker_secret": "regcred", "aws_secret": "aws-secret", "namespace": "saas-demo"}),
            "consul": OnlinePredictDocker(image="consul:1.13.1",
                                        ports=[ 8500 ], 
                                        options={"domain": "huawei.dmetasoul.com", "namespace": "saas-demo"}),
        }
    online_predict.services = {
        "mongo": OnlinePredictService(kind="MongoDB", serviceName="mongo", 
                                      collection=[scene.middle_result_storage.database], 
                                      options={"uri": 
                                          f"mongodb://{scene.middle_result_storage.user}:{scene.middle_result_storage.password}@{scene.middle_result_storage.host}:{scene.middle_result_storage.port}/{scene.middle_result_storage.database}?authSource={scene.middle_result_storage.authsource}"})
    }
    online_predict.source = OnlinePredictSource(
                                user_key_name="user_id",
                                item_key_name="item_id",
                                user_item_ids_name="user_bhv_item_seq",
                                user=OnlinePredictSourceItem(
                                    table=f"{scene.name}_user_feature", 
                                    serviceName="mongo", 
                                    collection=scene.middle_result_storage.database, 
                                    columns=[{"user_id": "str"}, {"user_bhv_item_seq": "str"}]),
                                item=OnlinePredictSourceItem(
                                    table=f"{scene.name}_item_feature", 
                                    serviceName="mongo", 
                                    collection=scene.middle_result_storage.database, 
                                    columns=[{"item_id": "str"}, {"brand": "str"}, {"category": "str"}]),
                                summary=OnlinePredictSourceItem(
                                    table=f"{scene.name}_item_summary", 
                                    serviceName="mongo", 
                                    collection=scene.middle_result_storage.database, 
                                    columns=[{"item_id": "str"}, {"brand": "str"}, {"category": "str"}, {"title": "str"}, {"description": "str"}, {"image": "str"}, {"url": "str"}, {"price": "str"}],
                                    max_reservation=scene.recommend_strategy.get("recommend_result_count")),
                                request=[{"user_id": "str"}, {"item_id": "str"}])
    online_predict.random_models = [OnlinePredictPopModel(name="pop", 
                                                        bound=10, 
                                                        recallService="recall_pop", 
                                                        keyName="key", 
                                                        valueName="value_list", 
                                                        source={"table": f"{scene.name}_pop", "serviceName": "mongo", "collection": "jpa"})]
    online_predict.cf_models = [OnlinePredictItemCFModel(name="swing",
                                                        recallService="recall_swing", 
                                                        relatedService="related_swing",
                                                        keyName="key", 
                                                        valueName="value", 
                                                        source={"table": f"{scene.name}_swing", "serviceName": "mongo", "collection": "jpa"})]
    online_predict.rank_models = [OnlinePredictDeepModel(name="widedeep",
                                                         model="widedeep",
                                                         rankService="rank_widedeep", 
                                                         column_info=
                                                         [{"dnn_sparse": ["user_id", "item_id", "brand", "category"]},
                                                          {"lr_sparse": ["user_id", "item_id", "category", "brand", "user_id#brand", "user_id#category"]}],
                                                         cross_features=[OnlinePredictRankModelCrossFetaure(name="user_id#brand",
                                                                                                            join="#", 
                                                                                                            fields=["user_id", "brand"]),
                                                                         OnlinePredictRankModelCrossFetaure(name="user_id#category",
                                                                                                            join="#",
                                                                                                            fields=["user_id", "category"])])]

    online_predict.experiments = [OnlinePredictExperiment(name="experiment.recall.swing", then=["recall_swing"]),
                                  OnlinePredictExperiment(name="experiment.related.swing", then=["related_swing"]),
                                  OnlinePredictExperiment(name="experiment.recall.pop", then=["recall_pop"]),
                                  OnlinePredictExperiment(name="experiment.rank.widedeep", then=["rank_widedeep"]),
                                  ]
    online_predict.layers = [OnlinePredictLayer(name="layer.recall", data={"experiment.recall.swing": 1.0}),
                             OnlinePredictLayer(name="layer.related", data={"experiment.related.swing": 1.0}),
                             OnlinePredictLayer(name="layer.rank", data={"experiment.rank.widedeep": 1.0})
                             ]
    online_predict.scenes = [OnlinePredictScene(name="guess-you-like", layers=["layer.recall", "layer.rank"], additionalRecalls=["pop"]),
                             OnlinePredictScene(name="looked-and-looked", layers=["layer.related", "layer.rank"], additionalRecalls=["pop"]),
                             ]
    algo.set_online_predict(online_predict=online_predict)
    algo.to_yaml()
    