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


from typing import Any, Dict, List, Optional

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
    

class LocalConfig(BaseModel):
    sharedVolumeInContainer: str = ""


class OfflineTrainingPythonTask(BaseModel):
    configPath: str
    scriptPath: str


class OfflineTraining(BaseModel):
    cronExpr: str = "* */6 * * *"
    dag: Dict = None


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
    fmt: str = "parquet"


class TrainingJobTraining(BaseModel):
    estimator_params: Optional[dict]


class DeepCTRModel(TrainingJobTraining):
    deep_ctr_model_class: Optional[str]
    model_config_class: Optional[str] 
    model_params: Optional[dict]
    estimator_config_class: Optional[str]


class ItemCFModel(TrainingJobTraining):
    i2i_estimator_class: Optional[str]
    i2i_estimator_config_class: Optional[str]


class TrainingJobMongo(BaseModel):
    uri: str
    database: str
    collection: Optional[str] = None
    write_mode: Optional[str] = "overwrite"
    index_fields: Optional[List[str]] = []
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
    fmt: Optional[str] = "parquet"

    
class TrainingJobJoinDataset(BaseModel):
    join_on: dict = None
    user_bhv_seq: dict = None
    negative_sample: dict = None


class TrainingJobSample(BaseModel):
    model_type: str = None
    split_test: float = None
    shuffle: bool = False
    fmt: str = "parquet"
    train_path: str = ""
    test_path: str = ""
    item_path: Optional[str] = ""
    combine_schema: Optional[Dict] = None


class TrainingJobFeatureTable(BaseModel):
    feature_column: Any = None
    mongo_collection: str = None
    index_fields: List[str] = None
    drop_duplicates_by: Optional[List[str]] = None
        
        
class TrainingJobFeature(BaseModel):
    mongodb: TrainingJobMongo = None
    tables: List[TrainingJobFeatureTable] = None


class JobResource(BaseModel):
    logging: TrainingJobLogging = TrainingJobLogging()
    spark: TrainingJobSparkResource = TrainingJobSparkResource()


class GenSampleResource(JobResource):
    load_dataset: TrainingJobSaveDataset = None
    join_dataset: Optional[TrainingJobJoinDataset] = None
    gen_feature: Optional[Dict] = {}
    gen_sample: Optional[List[TrainingJobSample]] = None
    dump_nn_feature: Optional[TrainingJobFeature] = None
    dump_lgb_feaure: Optional[TrainingJobFeature] = None


class TrainingJobResource(JobResource):
    load_dataset: TrainingJobLoadDataset = None
    save_dataset: TrainingJobSaveDataset = None
    dataset: TrainingJobDataset = TrainingJobDataset()
    training: TrainingJobTraining = TrainingJobTraining()
    mongodb: Optional[TrainingJobMongo] = None


class MetaSporeTask(BaseModel):
   name: str
   scriptPath: str
   resource: Optional[JobResource] = JobResource()
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
    columns: List[dict]
    max_reservation: Optional[int] = 200


class OnlinePredictSource(BaseModel):
    user_key_name: str
    item_key_name: str
    user_item_ids_name: str
    user_item_ids_split: str = "\\u0001"
    user: OnlinePredictSourceItem
    item: OnlinePredictSourceItem
    summary: OnlinePredictSourceItem
    request: List[dict]


class OnlinePredictModel(BaseModel):
    name: str
    

class OnlinePredictPopModel(OnlinePredictModel):
    bound: int
    recallService: str
    keyName: Optional[str]
    valueName: Optional[str]
    source: Optional[dict]


class OnlinePredictItemCFModel(OnlinePredictModel):
    recallService: str
    relatedService: str
    keyName: Optional[str]
    valueName: Optional[str]
    source: Optional[dict]


class OnlinePredictRankModelCrossFetaure(BaseModel):
    name: str
    join: str
    fields: List
    
    
class OnlinePredictDeepModel(OnlinePredictModel):
    model: str
    rankService: str
    column_info: List[Dict[str, List]]
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
    

class SceneDatasourceConnectInfo(BaseModel):
    type: str = "mysql"
    host: str = "127.0.0.1"
    port: int = 3306
    database: str = ""
    user: str = ""
    password: str = ""


class SceneDatasourceUserTableInfo(BaseModel):
    name: str = "user"
    user_id_column_name: str = "user_id"
    age_col_name: str = ""
    gender_col_name: str = ""
    city_col_name: str = ""
    

class SceneDatasourceItemTableInfo(BaseModel):
    name: str = "item"
    item_id_column_name: str = "item_id"
    price_column_name: str = ""
    title_column_name: str = ""
    brand_column_name: str = ""
    category_column_name: str = ""
    url_column_name: str = ""
    image_column_name: str = ""
    description_column_name: str = ""
    

class SceneDatasourceTransTableInfo(BaseModel):
    name: str = "interaction"
    user_id_column_name: str = "user_id"
    item_id_column_name: str = "item_id"
    timestamp_column_name: str = "timestamp"


class SceneDatasource(BaseModel):
    connect_info: SceneDatasourceConnectInfo = SceneDatasourceConnectInfo()
    user_tab_info: SceneDatasourceUserTableInfo = SceneDatasourceUserTableInfo()
    item_tab_info: SceneDatasourceItemTableInfo = SceneDatasourceItemTableInfo()
    trans_tab_info: SceneDatasourceTransTableInfo = SceneDatasourceTransTableInfo()


class SceneScedulerModel(BaseModel):
    name: str = SchedulerMode.LOCALMODE
    config: Any = LocalConfig()


class SceneMiddleResultStorage(BaseModel):
    type: str = "Mongo"
    host: str = "127.0.0.1"
    port: int = 27017
    database: str = ""
    user: str = ""
    password: str = ""
    authsource: Optional[str] = "admin"


class Scene(BaseModel):
    name: str
    scheduler_mode: SceneScedulerModel = SceneScedulerModel()
    datasource: SceneDatasource = SceneDatasource()
    middle_result_storage: SceneMiddleResultStorage = SceneMiddleResultStorage()
    recommend_strategy: Dict = {"recommend_result_count": 200}