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
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field


class SchedulerMode:
   LOCALMODE = "Local"
   K8SCLUSTER = "K8sCluster"
   SAGEMAKERMODE = "SageMaker"


class MetaSporeFlowKind:
   METASPOREFLOW = "MetaSporeFlow"
   SAGEMAKERCONFIG = "SageMakerConfig"
   OFFILINETRAINING = "OfflineSageMakerScheduler"
   OFFILINEPYTHONTASK= "OfflinePythonTask"
   TRAININGJOB = "TrainingJob"
   ONLINEPREDICT = "OnlineFlow"


class MetaSporeFlowMetadata(BaseModel):
    name: str
    category: Optional[str]
    uses: Optional[List] = list()


class MetaSporeFlowSpec(BaseModel):
    deployMode: str
    sharedVolumeInContainer: Optional[str] = Field(default=None)


class MetaSporeFlow(BaseModel):
    apiVersion: str = "metaspore/v1"
    kind: str = MetaSporeFlowKind.METASPOREFLOW
    metadata: MetaSporeFlowMetadata = MetaSporeFlowMetadata(name="metaspore_flow")
    spec: Any


class SageMakerConfig(BaseModel):
    roleArn: str = ""
    securityGroups: List = []
    subnets: List = []
    s3Endpoint: str = ""
    s3WorkDir: str = ""

class OfflineTrainingPythonTask(BaseModel):
    configPath: str
    scriptPath: str

class OfflineTrainingDag(BaseModel):
    sync_data: List = []
    join_data: List = []


class OfflineTraining(BaseModel):
    cronExpr: str = "* */6 * * *"
    dag: OfflineTrainingDag = None

class TrainingJobLogging(BaseModel):
    loglevel: str = "error"


class TrainingJobSparkSessionResource(BaseModel):
    app_name: str = "metaspore_training_job"
    local: str = "true"
    worker_count: int = 2
    worker_cpu: int = 1
    server_count: int = 2
    server_cpu: int = 1
    batch_size: int = 256
    worker_memory: str = "1G"
    server_memory: str = "1G"
    coordinator_memory: str = "1G"


class TrainingJobSparkResource(BaseModel):
    session_confs: TrainingJobSparkSessionResource = TrainingJobSparkSessionResource()
    extended_confs: dict = {}


class TrainingJobDataset(BaseModel):
    train_path: str = None
    test_path: str = None


class TrainingJobTraining(BaseModel):
    deep_ctr_model_class: Optional[str]
    i2i_estimator_class: Optional[str]
    i2i_estimator_config_class: Optional[str]
    model_config_class: Optional[str] 
    model_params: Optional[dict]
    estimator_class: Optional[str]
    estimator_config_class: Optional[str]
    estimator_params: Optional[dict]

class TrainingJobMongo(BaseModel):
    uri: str
    database: str
    collection: str
    write_mode: Optional[str] = "overwrite"
    index_fields: Optional[str] = []
    index_unique: Optional[bool] = False

class TrainingJobLoadDataset(BaseModel):
    format: str = "jdbc"
    driver: str = "com.mysql.jdbc.Driver"
    host: str = "127.0.0.1"
    port: int = 3306
    database: str = ""
    user: str = ""
    password: str = ""
    user_table: dict = None
    item_table: dict = None
    interaction_table: dict = None

class TrainingJobSaveDataset(BaseModel):
    user_path: str = None
    item_path: str = None
    interaction_path: str = None
    

class TrainingJobResource(BaseModel):
    logging: TrainingJobLogging = TrainingJobLogging()
    spark: TrainingJobSparkResource = TrainingJobSparkResource()
    load_dataset: TrainingJobLoadDataset = None
    save_dataset: TrainingJobSaveDataset = None
    dataset: TrainingJobDataset = TrainingJobDataset()
    training: TrainingJobTraining = TrainingJobTraining()
    mongodb: Optional[TrainingJobMongo] = None


class MetaSporeTask(BaseModel):
   name: str
   scriptPath: str
   resource: Optional[TrainingJobResource] = TrainingJobResource()
   category: Optional[str]
   
class OnlinePredictDocker(BaseModel):
    image: str
    ports: List[int]
    options: dict = None


class OnlinePredictService(BaseModel):
    kind: str
    serviceName: str
    collection: List[str]
    options: dict


class OnlinePredictSourceItem(BaseModel):
    table: str
    serviceName: str
    collection: str
    columns: dict


class OnlinePredictSource(BaseModel):
    user_key_name: str
    item_key_name: str
    user_item_ids_name: str
    user_item_ids_split: str = "\\u0001"
    user: OnlinePredictSourceItem
    item: OnlinePredictSourceItem
    summary: OnlinePredictSourceItem
    request: dict


class OnlinePredictModel(BaseModel):
    name: str
    bound: int
    recallService: str
    relatedService: Optional[str]
    keyName: Optional[str]
    valueName: Optional[str]
    source: Optional[dict]

class OnlinePredictRankModelCrossFetaure(BaseModel):
    name: str
    join: str
    fields: List


class OnlinePredictRankModel(BaseModel):
    name: str
    model: str
    rankService: str
    column_info: Dict[str, List]
    cross_features: List[OnlinePredictRankModelCrossFetaure]
    

class OnlinePredictExperiment(BaseModel):
    name: str
    then: List[str]
    

class OnlinePredictLayer(BaseModel):
    name: str
    data: dict


class OnlinePredictScene(BaseModel):
    name: str
    layers: List[str]
    additionalRecalls: List[str]


class OnlinePredict(BaseModel):
    dockers: Optional[Dict[str, OnlinePredictDocker]] = None
    services: Dict[str, OnlinePredictService] = None
    source: OnlinePredictSource = None
    random_models: List[OnlinePredictModel] = None
    cf_models: List[OnlinePredictModel] = None
    rank_models: List[OnlinePredictModel] = None
    experiments: List[OnlinePredictExperiment] = None
    layers: List[OnlinePredictLayer] = None
    scenes: List[OnlinePredictScene] = None


def extract_code(working_dir, codebase_tar='volumes.tar.gz'):
    scheduler_dir = os.path.dirname(os.path.abspath(__file__))
    with tarfile.open(os.path.join(scheduler_dir, codebase_tar), 'r:gz') as tfile:
        tfile.extractall(path=working_dir)
   

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
        self._metaspore.metadata.uses.append("./sage_maker_config.yaml")
        self._export_lists["sage_maker_config"] = _sagemakerconfig
        
    def set_offline_training(self, offline_training: OfflineTraining) -> None:
        self._offline_training = MetaSporeFlow(
            kind=MetaSporeFlowKind.OFFILINETRAINING,
            metadata=MetaSporeFlowMetadata(name=f"{self.scene_name}_offline_training"),
            spec=offline_training
        )
        self._metaspore.metadata.uses.append("./offline/offline_training.yaml")
        self._export_lists["offline/offline_training"] = self._offline_training
    
    def add_offline_training_dag(self, sync_data: List[MetaSporeTask] = None, 
                                 join_data: List[MetaSporeTask] = None) -> None:
        for task in sync_data:
            syc_data = MetaSporeFlow(
                        kind=MetaSporeFlowKind.OFFILINEPYTHONTASK,
                        metadata=MetaSporeFlowMetadata(name=f"{task.name}_task"),
                        spec=OfflineTrainingPythonTask(configPath=f"{task.name}.yaml", scriptPath=task.scriptPath), 
                    )
            self._offline_training.spec.dag.sync_data.append(task.name)
            self._export_lists[f"offline/task/{task.name}"] = syc_data
            self._offline_training.metadata.uses.append(f"./offline/task/{task.name}.yaml")
            
            res = MetaSporeFlow(
                    kind=MetaSporeFlowKind.TRAININGJOB,
                    metadata=MetaSporeFlowMetadata(name=f"{task.name}_res", category=task.category),
                    spec=task.resource, 
                )
            self._export_lists[f"volumes/{task.name}"] = res
            
        
        for task in join_data:
            join_data = MetaSporeFlow(
                        kind=MetaSporeFlowKind.OFFILINEPYTHONTASK,
                        metadata=MetaSporeFlowMetadata(name=f"{task.name}_task"),
                        spec=OfflineTrainingPythonTask(configPath=f"{task.name}.yaml", scriptPath=task.scriptPath),
                    )
            self._offline_training.spec.dag.join_data.append(task.name)
            self._export_lists[f"offline/task/{task.name}"] = join_data
            self._offline_training.metadata.uses.append(f"./offline/task/{task.name}.yaml")
            
            res = MetaSporeFlow(
                    kind=MetaSporeFlowKind.TRAININGJOB,
                    metadata=MetaSporeFlowMetadata(name=f"{task.name}_res", category=task.category),
                    spec=task.resource, 
                )
            self._export_lists[f"volumes/{task.name}"] = res
    
    def set_online_predict(self, online_predict: OnlinePredict) -> None:
        _online_predict = MetaSporeFlow(
            kind=MetaSporeFlowKind.ONLINEPREDICT,
            metadata=MetaSporeFlowMetadata(name=f"{self.scene_name}_online_predict"),
            spec=online_predict
        )
        self._metaspore.metadata.uses.append("./online/online_predict.yaml")
        self._export_lists["online/online_predict"] = _online_predict
    
    def to_yaml(self, saved_path: str="."):
        extract_code(f"{saved_path}/{self.scene_name}", 
                     codebase_tar="./volumes/sagemaker_volumes.tar.gz")
        for k, v in self._export_lists.items():
            if not os.path.exists(os.path.dirname(f"{saved_path}/{self.scene_name}/{k}.yaml")):
                os.makedirs(os.path.dirname(f"{saved_path}/{self.scene_name}/{k}.yaml"))
            with open(f"{saved_path}/{self.scene_name}/{k}.yaml", "w") as f:
                yaml.dump(v.dict(), f)


def generate_flow(scene_name: str, scheduler_mode: SchedulerMode):
    algo = GenerateFlow(scene_name, scheduler_mode)
    if scheduler_mode == SchedulerMode.SAGEMAKERMODE:
        algo.set_config(SageMakerConfig())
    algo.set_offline_training(OfflineTraining(
        cronExpr="* */6 * * *",
        dag=OfflineTrainingDag()
    ))
    algo.add_offline_training_dag(
        sync_data=[
            MetaSporeTask(name="setup", scriptPath="setup.py", category="Setup",
                          resource=TrainingJobResource(
                              spark=TrainingJobSparkResource(
                                  session_confs=TrainingJobSparkSessionResource(server_memory="13G")), 
                              load_dataset=TrainingJobLoadDataset(
                                  host="127.0.0.1", port="3306", database="metaspore_offline_flow", user="root", password="test_mysql_123456",
                                  user_table={"name": "user", "user_id_column_name": "user_id"}, 
                                  item_table={"name": "item", "item_id_column_name": "item_id", "price_column_name": "price",
                                              "title_column_name": "title", "brand_column_name": "brand", "category_column_name": "category", "url_column_name": "url", "description_column_name": "description"},
                                  interaction_table={"name": "interaction", "user_id_column_name": "user_id", "item_id_column_name": "item_id", "timestamp_column_name": "timestamp"}),
                              save_dataset=TrainingJobSaveDataset(
                                  user_path="@{METASPORE_FLOW_S3_WORK_DIR}/flow/scene/@{METASPORE_FLOW_SCENE_NAME}/data/raw/@{METASPORE_FLOW_MODEL_VERSION}/amazon_fashion_user.parquet",
                                  item_path="@{METASPORE_FLOW_S3_WORK_DIR}/flow/scene/@{METASPORE_FLOW_SCENE_NAME}/data/raw/@{METASPORE_FLOW_MODEL_VERSION}/amazon_fashion_item.parquet",
                                  interaction_path="@{METASPORE_FLOW_S3_WORK_DIR}/flow/scene/@{METASPORE_FLOW_SCENE_NAME}/data/raw/@{METASPORE_FLOW_MODEL_VERSION}/amazon_fashion_interaction.parquet"))),
            MetaSporeTask(name="gen_samples", scriptPath="gen_samples.py", category="Dataset/SampleGeneration",
                          resource=TrainingJobResource(spark=TrainingJobSparkResource(session_confs=TrainingJobSparkSessionResource(server_memory="16G"))))],
        join_data=[
            MetaSporeTask(name="popular_topk", scriptPath="popular_topk.py", category="Match/Popular",
                          resource=TrainingJobResource(training=TrainingJobTraining(
                              estimator_params={"max_recommendation_count": 20, "group_nums": 10}),
                                                       mongodb=TrainingJobMongo(
                                                           uri="mongodb://root:test_mongodb_123456@172.31.47.204:57018/?authSource=admin",
                                                           database="jpa", collection="amazonfashion_swing"))),
            MetaSporeTask(name="itemcf", scriptPath="itemcf.py", category="Match/I2I/Swing",
                          resource=TrainingJobResource(logging=TrainingJobLogging(loglevel="debug"),
                                                       dataset=TrainingJobDataset(train_path="@{METASPORE_FLOW_S3_WORK_DIR}/flow/scene/@{METASPORE_FLOW_SCENE_NAME}/data/ctr/nn/@{METASPORE_FLOW_MODEL_VERSION}/train.parquet", test_path="@{METASPORE_FLOW_S3_WORK_DIR}/flow/scene/@{METASPORE_FLOW_SCENE_NAME}/data/ctr/nn/@{METASPORE_FLOW_MODEL_VERSION}/test.parquet"),
                                                       training=TrainingJobTraining(
                                                           i2i_estimator_class="metaspore.algos.item_cf_retrieval.ItemCFEstimator",
                                                           i2i_estimator_config_class="metaspore.algos.pipeline.ItemCFEstimatorConfig",
                                                           estimator_params={"max_recommendation_count": 20}),
                                                       mongodb=TrainingJobMongo(
                                                           uri="mongodb://root:test_mongodb_123456@172.31.47.204:57018/?authSource=admin",
                                                           database="jpa", collection="amazonfashion_swing"))),
            MetaSporeTask(name="deepctr", scriptPath="deepctr.py", category="Rank/DeepCTR/WideDeep", 
                          resource=TrainingJobResource(spark=TrainingJobSparkResource(extended_confs={
                                "spark.network.timeout": "500",
                                "spark.ui.showConsoleProgress": "true"
                          }),training=TrainingJobTraining(
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
    if scheduler_mode == SchedulerMode.LOCALMODE:
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
        "mongo": OnlinePredictService(kind="MongoDB", serviceName="mongo", collection=["jpa"], 
                                      options={"uri": "mongodb://root:test_mongodb_123456@$${MONGO_HOST:172.31.47.204}:$${MONGO_PORT:57018}/jpa?authSource=admin"})
    }
    online_predict.source = OnlinePredictSource(
                                user_key_name="user_id",
                                item_key_name="item_id",
                                user_item_ids_name="user_bhv_item_seq",
                                user=OnlinePredictSourceItem(
                                    table="amazonfashion_user_feature", 
                                    serviceName="mongo", 
                                    collection="jpa", 
                                    columns={"user_id": "str", "user_bhv_item_seq": "str"}),
                                item=OnlinePredictSourceItem(
                                    table="amazonfashion_item_feature", 
                                    serviceName="mongo", 
                                    collection="jpa", 
                                    columns={"item_id": "str", "brand": "str", "category": "str"}),
                                summary=OnlinePredictSourceItem(
                                    table="amazonfashion_item_summary", 
                                    serviceName="mongo", 
                                    collection="jpa", 
                                    columns={"item_id": "str", "brand": "str", "category": "str", "title": "str", "description": "str", "image": "str", "url": "str", "image": "price"}),
                                request={"user_id": "str", "item_id": "str"})
    online_predict.random_models = [OnlinePredictModel(name="pop", 
                                                       bound=10, 
                                                       recallService="recall_pop", 
                                                       keyName="key", 
                                                       valueName="value_list", 
                                                       source={"table": "amazonfashion_pop", "serviceName": "mongo", "collection": "jpa"})]
    online_predict.cf_models = [OnlinePredictModel(name="swing", 
                                                   bound=10, 
                                                   recallService="recall_swing", 
                                                   relatedService="related_swing",
                                                   keyName="key", 
                                                   valueName="value_list", 
                                                   source={"table": "amazonfashion_swing", "serviceName": "mongo", "collection": "jpa"})]
    online_predict.rank_models = [OnlinePredictRankModel(name="widedeep",
                                                         model="widedeep",
                                                         rankService="rank_widedeep", 
                                                         column_info={
                                                             "dnn_sparse": ["user_id", "item_id", "brand", "category"],
                                                             "lr_sparse": ["user_id", "item_id", "category", "brand", "user_id#brand", "user_id#category"]},
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
    
    