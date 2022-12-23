sync_data_template = """
apiVersion: metaspore/v1
kind: OfflinePythonTask
metadata:
  name: sync_data
spec:
  scriptPath: "setup.py"
  configPath: "setup.yaml"
"""

crontab_scheduler_template = """
apiVersion: metaspore/v1
kind: OfflineCrontabScheduler
metadata:
  name: offline_crontab_scheduler
spec:
  cronExpr: "*/20 * * * *"
  dag:
    sync_data: [ "join_data" ]
    join_data: [ ]
"""

offline_flow_template = """
apiVersion: metaspore/v1
kind: OfflineLocalFlow
metadata:
  name: offline_local_flow
spec:
  offlineLocalImage: 'swr.cn-southwest-2.myhuaweicloud.com/dmetasoul-repo/metaspore-training-release:v1.1.0'
  offlineLocalContainerName: 'metaspore_offline_flow'
"""

metaspore_flow_template = """
apiVersion: metaspore/v1
kind: MetaSporeFlow
metadata:
    name: metaspore_flow
    uses:
spec:
    sharedVolumeInContainer: '/opt/volumes'
"""

online_flow_template = r"""
apiVersion: metaspore/v1
kind: OnlineFlow
metadata:
  name: online_local_flow
spec:
  dockers:
    recommend:
      image: swr.cn-southwest-2.myhuaweicloud.com/dmetasoul-repo/recommend-service:1.0.0
      ports: [ 13013 ]
      options:
        mongo_service: 192.168.0.22
        mongo_port: 27018
    model:
      image: swr.cn-southwest-2.myhuaweicloud.com/dmetasoul-repo/metaspore-serving-release:cpu-v1.0.1
      ports: [ 50000 ]
      options:
        watch_image: swr.cn-southwest-2.myhuaweicloud.com/dmetasoul-repo/consul-watch-load:v1.0.0
        watch_port: 8080
        consul_key: dev/
        endpoint_url: http://obs.cn-southwest-2.myhuaweicloud.com
        docker_secret: regcred
        aws_secret: aws-secret
    consul:
      image: consul:1.13.1
      ports: [ 8500 ]
  services:
    mongo:
      kind: MongoDB
      serviceName: mongo
      collection: [ jpa ]
      options:
        uri: "mongodb://root:test_mongodb_123456@$${MONGO_HOST:172.17.0.1}:$${MONGO_PORT:27018}/jpa?authSource=admin"
  source:
    user_key_name: user_id
    item_key_name: item_id
    user_item_ids_name: user_bhv_item_seq
    user_item_ids_split: "\u0001"
    user:
      table: amazonfashion_user_feature
      serviceName: mongo
      collection: jpa
      columns:
        - user_id: str
        - user_bhv_item_seq: str
    item:
      table: amazonfashion_item_feature
      serviceName: mongo
      collection: jpa
      columns:
        - item_id: str
        - brand: str
        - category: str
    summary:
      table: amazonfashion_item_summary
      serviceName: mongo
      collection: jpa
      columns:
        - item_id: str
        - brand: str
        - category: str
        - title: str
        - description: str
        - image: str
        - url: str
        - price: str
    request:
      - user_id: str
      - item_id: str
  random_models:
    - name: pop
      bound: 10
      keyName: key
      valueName: value_list
      source:
        table: amazonfashion_pop
        serviceName: mongo
        collection: jpa
  cf_models:
    - name: swing
      keyName: key
      valueName: value
      source:
        table: amazonfashion_swing
        serviceName: mongo
        collection: jpa
  rank_models:
    - name: widedeep
      model: amazonfashion_widedeep
      column_info:
        - dnn_sparse: [ "user_id", "item_id", "brand", "category" ]
        - lr_sparse: [ "user_id", "item_id", "category", "brand", "user_id#brand", "user_id#category" ]
      cross_features:
        - name: user_id#brand
          join: "#"
          fields: [ "user_id", "brand" ]
        - name: user_id#category
          join: "#"
          fields: [ "user_id", "category" ]
  experiments:
    - name: experiment.recall.swing
      then: [ recall_swing ]
    - name: experiment.related.swing
      then: [ related_swing ]
    - name: experiment.rank.widedeep
      then: [ rank_widedeep ]
  layers:
    - name: layer.recall
      data:
        experiment.recall.swing: 1.0
    - name: layer.related
      data:
        experiment.related.swing: 1.0
    - name: layer.rank
      data:
        experiment.rank.widedeep: 1.0
  scenes:
    - name: guess-you-like
      layers: [ layer.recall, layer.rank ]
      additionalRecalls: [ pop ]
    - name: looked-and-looked
      layers: [ layer.related, layer.rank ]
      additionalRecalls: [ pop ]
"""

join_data_template = """
apiVersion: metaspore/v1
kind: OfflinePythonTask
metadata:
  name: join_data
spec:
  scriptPath: "gen_samples.py"
  configPath: "gen_samples.yaml"
"""

train_model_pop_template = """
apiVersion: metaspore/v1
kind: OfflinePythonTask
metadata:
  name: train_model_pop
spec:
  scriptPath: "popular_topk.py"
  configPath: "popular_topk.yaml"
"""

train_model_icf_template = """
apiVersion: metaspore/v1
kind: OfflinePythonTask
metadata:
  name: train_model_itemcf
spec:
  scriptPath: "itemcf.py"
  configPath: "swing_i2i.yaml"
"""

train_model_ctr_template = """
apiVersion: metaspore/v1
kind: OfflinePythonTask
metadata:
  name: train_model_deepctr
spec:
  scriptPath: "deepctr.py"
  configPath: "deepctr.yaml"
"""

notify_load_model_template = """
apiVersion: metaspore/v1
kind: OfflinePythonTask
metadata:
  name: notify_load_model
spec:
  scriptPath: "notify_load_model.py"
  configPath: "deepctr.yaml"
"""
