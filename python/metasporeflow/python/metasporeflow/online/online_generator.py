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
from .cloud_consul import putServiceConfig
from .common import DumpToYaml, dictToObj, setDefault, S
from .compose_config import OnlineDockerCompose
from .online_flow import OnlineFlow, ServiceInfo, DataSource, FeatureInfo, CFModelInfo, RankModelInfo, DockerInfo, \
    RandomModelInfo, CrossFeature
from .service_config import get_source_option, Source, Condition, FieldAction, \
    FeatureConfig, RecommendConfig, TransformConfig, Chain, ExperimentItem, OnlineServiceConfig


def append_source_table(feature_config, name, datasource, default_columns=[]):
    if feature_config is None or not datasource:
        raise ValueError("datasource must set!")
    source_name = datasource.serviceName
    if datasource.collection:
        source_name = "%s_%s" % (datasource.serviceName, datasource.collection)
    if not feature_config.find_source(source_name):
        raise ValueError("source: %s must set in services!" % source_name)
    columns = setDefault(datasource, "columns", default_columns)
    if not columns:
        raise ValueError("ds columns must not be empty")
    feature_config.add_sourceTable(name=name, source=source_name, table=datasource.table,
                                   columns=columns)


def columns_has_key(columns, key):
    if not columns or not key:
        return False
    for field in columns:
        if key in field:
            return True
    return False


class OnlineGenerator(object):
    def __init__(self, **kwargs):
        self.resource = kwargs.get("resource")
        if not self.resource or not isinstance(self.resource.data, OnlineFlow):
            raise ValueError("MetaSpore Online need input online configure data!")
        self.local_resource = kwargs.get("local_resource")
        if not self.local_resource or not self.local_resource.data:
            raise ValueError("MetaSpore need input configure data!")
        self.local_config = self.local_resource.data
        self.configure = self.resource.data

    def gen_docker_compose(self):
        online_docker_compose = OnlineDockerCompose()
        dockers = {}
        if self.configure.dockers:
            dockers.update(self.configure.dockers)
        if "recommend" not in dockers:
            dockers["recommend"] = DockerInfo("swr.cn-southwest-2.myhuaweicloud.com/dmetasoul-repo/recommend-service-11:1.0.14", {})
        no_mode_service = True
        for name in dockers.keys():
            if str(name).startswith("model"):
                no_mode_service = False
                break
        if no_mode_service:
            dockers["model"] = \
                DockerInfo("swr.cn-southwest-2.myhuaweicloud.com/dmetasoul-repo/metaspore-serving-release:cpu-v1.0.1",
                           {})
        for name, info in dockers.items():
            info = dictToObj(info)
            volumes = []
            for key, value in info.volumes.items():
                volumes.append("%s/volumes/%s" % (os.getcwd(), value))
            if not volumes and str(name).startswith("model"):
                volumes.append("%s/volumes/output/model:/data/models" % os.getcwd())
            online_docker_compose.add_service(name, "container_%s_service" % name,
                                              image=info.image, volumes=volumes, ports=info.ports)
        online_recommend_service = online_docker_compose.services.get("recommend")
        if not online_recommend_service:
            raise ValueError("container_recommend_service init fail!")
        if online_docker_compose.services:
            for name, service in online_docker_compose.services.items():
                service = dictToObj(service)
                if name == "recommend" or not service.ports:
                    continue
                online_recommend_service.add_env("%s_HOST" % name.upper(), name)
                online_recommend_service.add_env("%s_PORT" % name.upper(), service.ports[0])
        return online_docker_compose

    def gen_k8s_config(self):
        consul_data = {}
        recommend_data = {}
        model_data = {}
        if not self.configure.dockers:
            return consul_data, recommend_data, model_data
        for name, info in self.configure.dockers.items():
            info = dictToObj(info)
            data = {}
            if "volumes" in info:
                for key, value in info.volumes.items():
                    data["volume_%s" % key] = value
            if "ports" in info:
                data["port"] = info.ports[0]
            if "options" in info:
                data.update(info.options)
            if "image" in info:
                data["image"] = info.image
            data["name"] = "%s-k8s-service" % name
            if name == "recommend":
                recommend_data.update(data)
            elif name == "consul":
                consul_data.update(data)
            elif name == "model":
                model_data.update(data)
        recommend_data["consul_port"] = consul_data.setdefault("port", 8500)
        model_data["consul_port"] = consul_data.setdefault("port", 8500)
        recommend_data["consul_service"] = consul_data["name"]
        model_data["consul_service"] = consul_data["name"]
        recommend_data["model_service"] = model_data["name"]
        recommend_data["model_port"] = model_data["port"]
        return consul_data, recommend_data, model_data

    def process_feature_source(self, feature_config):
        if not self.configure.services:
            raise ValueError("services must set!")
        for name, info in self.configure.services.items():
            info = dictToObj(info)
            if "collection" not in info:
                feature_config.add_source(name=name, kind=info.kind,
                                          options=get_source_option(self.configure, name, None))
            else:
                for db in info.collection:
                    feature_config.add_source(name="%s_%s" % (name, db), kind=info.kind,
                                              options=get_source_option(self.configure, name, db))

    def init_feature_info(self, feature_info):
        self.user_key = feature_info.user_key_name or "user_id"
        self.items_key = feature_info.user_item_ids_name or "user_bhv_item_seq"
        self.item_key = feature_info.item_key_name or "item_id"
        if not columns_has_key(feature_info.user.columns, self.user_key) \
                or not columns_has_key(feature_info.user.columns, self.items_key):
            raise ValueError("user column must has user_key_name and user_item_ids_name!")
        if not columns_has_key(feature_info.item.columns, self.item_key):
            raise ValueError("item column must has item_key_name!")
        if not columns_has_key(feature_info.summary.columns, self.item_key):
            raise ValueError("summary column must has item_key_name!")
        self.request_columns = feature_info.request
        if not self.request_columns:
            self.request_columns = [{self.user_key: "str"}, {self.item_key: "str"}]
        if not columns_has_key(self.request_columns, self.user_key):
            raise ValueError("request column must set user_key!")
        if not columns_has_key(self.request_columns, self.item_key):
            raise ValueError("request column must set item_key!")

        self.user_fields = list()
        self.user_key_type = "str"
        for field_item in feature_info.user.columns:
            self.user_fields.extend(field_item.keys())
            if self.user_key in field_item:
                self.user_key_type = field_item.get(self.user_key)
        self.request_fields = list()
        for field_item in self.request_columns:
            self.request_fields.extend(field_item.keys())
            if self.user_key in field_item and field_item.get(self.user_key) != self.user_key_type:
                raise ValueError("request user key type set error!")
        self.item_fields = list()
        self.item_key_type = "str"
        for field_item in feature_info.item.columns:
            self.item_fields.extend(field_item.keys())
            if self.item_key in field_item:
                self.item_key_type = field_item.get(self.item_key)
        self.summary_fields = list()
        for field_item in feature_info.summary.columns:
            self.summary_fields.extend(field_item.keys())
            if self.item_key in field_item:
                if self.item_key_type != field_item.get(self.item_key):
                    raise ValueError("item and summary item key type set error!")

    def process_feature_sourceTable(self, feature_config):
        feature_info = self.configure.source
        if not feature_info:
            raise ValueError("feature_info must set!")
        feature_info = dictToObj(feature_info)
        self.init_feature_info(feature_info)
        append_source_table(feature_config, "source_table_user", feature_info.user)
        append_source_table(feature_config, "source_table_item", feature_info.item)
        append_source_table(feature_config, "source_table_summary", feature_info.summary)
        feature_config.add_sourceTable(
            name="source_table_request", source="request", columns=self.request_columns)

    def process_expriments(self, service_dict, recommend_config):
        recommend_experiments = set()
        if self.configure.experiments:
            for data in self.configure.experiments:
                experiment_data = dictToObj(data)
                then = []
                when = []
                if "then" in experiment_data:
                    for model_name in experiment_data.then:
                        if model_name in service_dict:
                            then.append(service_dict.get(model_name))
                        else:
                            print("no found service from model name: %s" % model_name)
                if "when" in experiment_data:
                    for model_name in experiment_data.when:
                        if model_name in service_dict:
                            when.append(service_dict.get(model_name))
                        else:
                            print("no found service from model name: %s" % model_name)
                recommend_config.add_experiment(name=experiment_data.name,
                                                options={"maxReservation": 200}, chains=[
                        Chain(then=then, when=when, transforms=[
                            TransformConfig(name="cutOff"),
                            TransformConfig(name="updateField", option={
                                "input": ["score", "origin_scores"], "output": ["origin_scores"],
                                "updateOperator": "putOriginScores"
                            })
                        ])
                    ])
                recommend_experiments.add(experiment_data.name)
        return recommend_experiments

    def process_layers(self, recommend_experiments, recommend_config):
        layer_set = set()
        if self.configure.layers:
            for data in self.configure.layers:
                layer_config = dictToObj(data)
                layer_name = layer_config.name
                experiments = list()
                for experiment_name, rato in layer_config.data.items():
                    if experiment_name not in recommend_experiments:
                        print("no experiment_name:%s found" % experiment_name)
                        continue
                    experiments.append(ExperimentItem(name=experiment_name, ratio=rato))
                recommend_config.add_layer(name=layer_name, bucketizer="random", experiments=experiments)
                layer_set.add(layer_name)
        return layer_set

    def process_scenes(self, layer_set, feature_config, recommend_config):
        iteminfo_service_name = "iteminfo_summary"
        iteminfo_columns = [{self.user_key: self.user_key_type},
                            {self.item_key: self.item_key_type},
                            {"score": "double"},
                            {"origin_scores": "map_str_double"},
                            ]
        feature_iteminfo_name = "feature_itemInfo_summary"
        recommend_config.add_service(name=iteminfo_service_name,
                                     preTransforms=[TransformConfig(name="summary")],
                                     columns=iteminfo_columns,
                                     tasks=[feature_iteminfo_name], options={"maxReservation": 200})
        summary_select = ["source_table_summary.%s" % field for field in self.summary_fields]
        for field_item in iteminfo_columns:
            summary_select.extend(["%s.%s" % (iteminfo_service_name, field) for field in field_item.keys() if
                                   field != self.user_key and field != self.item_key])
        feature_config.add_feature(name=feature_iteminfo_name, depend=[iteminfo_service_name, "source_table_summary"],
                                   select=summary_select,
                                   condition=[
                                       Condition(left="%s.%s" % (iteminfo_service_name, self.item_key), type="left",
                                                 right="source_table_summary.%s" % self.item_key)])
        random_recall_dict = self.add_random_recalls(feature_config)
        if self.configure.scenes:
            for data in self.configure.scenes:
                scene_data = dictToObj(data)
                layers = []
                for layer_name in scene_data.layers:
                    if layer_name not in layer_set:
                        print("no layer_name:%s found" % layer_name)
                        continue
                    layers.append(layer_name)
                transforms = [
                    TransformConfig(name="cutOff", option={
                        "dupFields": [self.user_key, self.item_key],
                        "or_filter_data": "source_table_request",
                        "or_field_list": [self.item_key],
                    }),
                    TransformConfig(name="updateField", option={
                        "input": ["score", "origin_scores"], "output": ["origin_scores"],
                        "updateOperator": "putOriginScores"
                    }),
                ]
                if "additionalRecalls" in scene_data and scene_data.additionalRecalls:
                    if not random_recall_dict:
                        raise ValueError("random_recall_model set empty!")
                    for model_name in scene_data.additionalRecalls:
                        if model_name not in random_recall_dict:
                            raise ValueError("random_recall_model: %s not set in config!" % model_name)
                    transforms.append(TransformConfig(name="additionalRecall", option={
                        "recall_list": [random_recall_dict[item] for item in scene_data.additionalRecalls],
                        "min_request": 10
                    }))
                transforms.append(TransformConfig(name="addItemInfo", option={
                    "service_name": iteminfo_service_name,
                }))
                recommend_config.add_scene(name=scene_data.name, chains=[Chain(then=layers, transforms=transforms)],
                                           columns=[{self.user_key: self.user_key_type},
                                                    {self.item_key: self.item_key_type}])

    def add_item_summary(self, feature_config):
        feature_config.add_feature(name="feature_item_summary", depend=["source_table_request", "source_table_summary"],
                                   select=["source_table_summary.%s" % field for field in self.summary_fields],
                                   condition=[Condition(left="source_table_request.%s" % self.item_key, type="left",
                                                        right="source_table_summary.%s" % self.item_key)])

    def add_user_profile(self, feature_config):
        feature_info = self.configure.source
        if not feature_info:
            raise ValueError("feature_info must set!")
        feature_info = dictToObj(feature_info)
        feature_config.add_feature(name="feature_user", depend=["source_table_request", "source_table_user"],
                                   select=["source_table_user.%s" % field for field in self.user_fields],
                                   condition=[Condition(left="source_table_request.%s" % self.user_key, type="left",
                                                        right="source_table_user.%s" % self.user_key)])
        user_key_action = FieldAction(names=["typeTransform.%s" % self.user_key], types=["str"], fields=[self.user_key],
                                      func="typeTransform", )
        user_profile_actions = list([user_key_action, ])
        user_profile_actions.append(FieldAction(names=["item_ids"], types=["list_str"],
                                                options={"splitor": feature_info.user_item_ids_split or "\u0001"},
                                                func="splitRecentIds", fields=[self.items_key]))
        user_profile_actions.append(
            FieldAction(names=[self.user_key, self.item_key, "item_score"], types=["str", "str", "double"],
                        func="recentWeight", input=["typeTransform.%s" % self.user_key, "item_ids"]))
        feature_config.add_algoTransform(name="algotransform_user", taskName="UserProfile", feature=["feature_user"],
                                         fieldActions=user_profile_actions,
                                         output=[self.user_key, self.item_key, "item_score"])

    def add_random_recalls(self, feature_config):
        random_recall_dict = dict()
        if self.configure.random_models:
            model_info = self.configure.random_models[0]
            model_info = dictToObj(model_info)
            if not model_info.name:
                raise ValueError("random_model model name must not be empty")
            random_recall = self.add_one_random_recall(model_info, feature_config)
            if random_recall is not None:
                random_recall_dict[model_info.name] = random_recall
        return random_recall_dict

    def add_one_random_recall(self, model_info, feature_config):
        if model_info:
            key_name = model_info.keyName
            value_name = model_info.valueName
            if "columns" not in model_info.source:
                columns = [{key_name: "int"}, {value_name: {"list_struct": {"item_id": "str", "score": "double"}}}]
            else:
                columns = model_info.source.columns
            if not columns_has_key(columns, key_name):
                raise ValueError("random model datasource column must has key_name: %s!" % key_name)
            if not columns_has_key(columns, value_name):
                raise ValueError("random model datasource column must has value_name: %s!" % value_name)
            append_source_table(feature_config, model_info.name, model_info.source, columns)
            random_bound = model_info.bound
            if random_bound <= 0:
                random_bound = 10
            random_task_name_1 = "algotransform_random_%s" % model_info.name
            feature_config.add_algoTransform(name=random_task_name_1,
                                             fieldActions=[FieldAction(names=["hash_id"], types=["int"],
                                                                       func="randomGenerator",
                                                                       options={"bound": random_bound}),
                                                           FieldAction(names=["score"], types=["double"],
                                                                       func="setValue",
                                                                       options={"value": 1.0})
                                                           ],
                                             output=["hash_id", "score"])
            random_task_name_2 = "feature_random_%s" % model_info.name
            feature_config.add_feature(name=random_task_name_2,
                                       depend=["source_table_request", random_task_name_1, model_info.name],
                                       select=["source_table_request.%s" % self.user_key,
                                               "%s.score" % random_task_name_1,
                                               "%s.%s" % (model_info.name, value_name)],
                                       condition=[Condition(left="%s.hash_id" % random_task_name_1,
                                                            right="%s.%s" % (model_info.name, key_name))])
            field_actions = list()
            field_actions.append(FieldAction(names=["toItemScore.%s" % self.user_key, "item_score"],
                                             types=["str", "map_str_double"],
                                             func="toItemScore", fields=[self.user_key, value_name, "score"]))
            field_actions.append(
                FieldAction(names=[self.user_key, self.item_key, "score", "origin_scores"],
                            types=["str", "str", "double", "map_str_double"],
                            func="recallCollectItem", input=["toItemScore.%s" % self.user_key, "item_score"]))
            random_task_name_3 = "algotransform_recall_%s" % model_info.name
            feature_config.add_algoTransform(name=random_task_name_3,
                                             taskName="ItemMatcher", feature=[random_task_name_2],
                                             options={"algo-name": model_info.name},
                                             fieldActions=field_actions,
                                             output=[self.user_key, self.item_key, "score", "origin_scores"])
            return random_task_name_3
        return None

    def add_cf_recalls(self, service_dict, feature_config, recommend_config):
        if self.configure.cf_models:
            for model_info in self.configure.cf_models:
                model_info = dictToObj(model_info)
                self.add_one_cf_recall(model_info, service_dict, feature_config, recommend_config)

    def add_one_cf_recall(self, model_info, service_dict, feature_config, recommend_config):
        if model_info:
            if not model_info.name:
                raise ValueError("cf_model model name must not be empty")
            key_name = model_info.keyName
            value_name = model_info.valueName
            if "columns" not in model_info.source:
                columns = [{key_name: "str"}, {value_name: {"list_struct": {"_1": "str", "_2": "double"}}}]
            else:
                columns = model_info.source.columns
            if not columns_has_key(columns, key_name):
                raise ValueError("cf model datasource column must has key_name: %s!" % key_name)
            for info in columns:
                if key_name in info:
                    assert info[key_name] == "str"
            if not columns_has_key(columns, value_name):
                raise ValueError("cf model datasource column must has value_name: %s!" % value_name)
            append_source_table(feature_config, model_info.name, model_info.source, columns)
            cf_task_name_1 = "feature_cf_{}".format(model_info.name)
            feature_config.add_feature(name=cf_task_name_1, depend=["algotransform_user", model_info.name],
                                       select=["algotransform_user.%s" % self.user_key, "algotransform_user.item_score",
                                               "%s.value" % model_info.name],
                                       condition=[Condition(left="algotransform_user.%s" % self.item_key, type="left",
                                                            right="%s.key" % model_info.name)])
            related_cf_task_name_1 = "feature_related_cf_{}".format(model_info.name)
            feature_config.add_feature(name=related_cf_task_name_1, depend=["source_table_request", model_info.name],
                                       select=["source_table_request.%s" % self.user_key,
                                               "%s.value" % model_info.name],
                                       condition=[Condition(left="source_table_request.%s" % self.item_key, type="left",
                                                            right="%s.key" % model_info.name)])
            field_actions = [FieldAction(names=["toItemScore.{}".format(self.user_key), "item_score"],
                                         types=["str", "map_str_double"],
                                         func="toItemScore", fields=[self.user_key, "value", "item_score"]),
                             FieldAction(names=[self.user_key, self.item_key, "score", "origin_scores"],
                                         types=["str", "str", "double", "map_str_double"],
                                         func="recallCollectItem",
                                         input=["toItemScore.{}".format(self.user_key), "item_score"])]
            cf_task_name_2 = "algotransform_%s" % model_info.name
            feature_config.add_algoTransform(name=cf_task_name_2,
                                             taskName="ItemMatcher", feature=[cf_task_name_1],
                                             options={"algo-name": model_info.name},
                                             fieldActions=field_actions,
                                             output=[self.user_key, self.item_key, "score", "origin_scores"])
            related_field_actions = [FieldAction(names=["toItemScore.{}".format(self.user_key), "item_score"],
                                                 types=["str", "map_str_double"], input=["setValue.item_score"],
                                                 func="toItemScore", fields=[self.user_key, "value"]),
                                     FieldAction(names=[self.user_key, self.item_key, "score", "origin_scores"],
                                                 types=["str", "str", "double", "map_str_double"],
                                                 func="recallCollectItem",
                                                 input=["toItemScore.{}".format(self.user_key), "item_score"]),
                                     FieldAction(names=["setValue.item_score"], types=["double"],
                                                 func="setValue",
                                                 options={"value": 1.0})]
            related_cf_task_name_2 = "algotransform_related_%s" % model_info.name
            feature_config.add_algoTransform(name=related_cf_task_name_2,
                                             taskName="ItemMatcher", feature=[related_cf_task_name_1],
                                             options={"algo-name": model_info.name},
                                             fieldActions=related_field_actions,
                                             output=[self.user_key, self.item_key, "score", "origin_scores"])
            service_name = "recall_%s" % model_info.name
            recommend_config.add_service(name=service_name, tasks=[cf_task_name_2],
                                         options={"maxReservation": 200})
            service_dict[service_name] = service_name
            related_service_name = "related_%s" % model_info.name
            recommend_config.add_service(name=related_service_name, tasks=[related_cf_task_name_2],
                                         options={"maxReservation": 200})
            service_dict[related_service_name] = related_service_name

    def add_rank_models(self, service_dict, feature_config, recommend_config):
        if self.configure.rank_models:
            for model_info in self.configure.rank_models:
                model_info = dictToObj(model_info)
                self.add_one_rank_recall(model_info, service_dict, feature_config, recommend_config)

    def add_one_rank_recall(self, model_info, service_dict, feature_config, recommend_config):
        if model_info:
            if not model_info.name or not model_info.model:
                raise ValueError("rank_models model name or model must not be empty")
            rank_task_name_1 = "feature_rank_%s" % model_info.name
            service_name = "rank_%s" % model_info.name
            select_fields = list()
            select_fields.extend(["source_table_user.%s" % key for key in self.user_fields])
            select_fields.extend(["source_table_item.%s" % key for key in self.item_fields])
            select_fields.append("rank_%s.origin_scores" % model_info.name)
            feature_config.add_feature(name=rank_task_name_1,
                                       depend=["source_table_user", "source_table_item",
                                               service_name],
                                       select=select_fields,
                                       condition=[Condition(left="source_table_user.%s" % self.user_key,
                                                            right="%s.%s" % (service_name, self.user_key)),
                                                  Condition(left="source_table_item.%s" % self.item_key,
                                                            right="%s.%s" % (service_name, self.item_key)),
                                                  ])
            field_actions = list()
            cross_features = list()
            if model_info.cross_features:
                for cross_item in model_info.cross_features:
                    cross_item = dictToObj(cross_item)
                    cross_features.append(cross_item.name)
                    field_actions.append(FieldAction(names=[cross_item.name],
                                                     types=["str"],
                                                     func="concatField", options={"join": cross_item.join},
                                                     fields=cross_item.fields))
            field_actions.append(FieldAction(names=[self.user_key, "typeTransform.%s" % self.item_key],
                                             types=["str", "str"],
                                             func="typeTransform",
                                             fields=[self.user_key, self.item_key]))
            field_actions.append(FieldAction(names=[self.item_key, "score", "origin_scores"],
                                             types=["str", "float", "map_str_double"],
                                             input=["typeTransform.%s" % self.item_key, "rankScore"],
                                             func="rankCollectItem",
                                             fields=["origin_scores"]))
            column_info = model_info.column_info
            if not column_info:
                column_info = [{"dnn_sparse": [self.item_key]}, {"lr_sparse": [self.item_key]}]
            algo_inputs = ["typeTransform.%s" % self.item_key]
            algo_inputs.extend(cross_features)
            field_actions.append(FieldAction(names=["rankScore"], types=["float"],
                                             algoColumns=column_info,
                                             options={"modelName": model_info.model,
                                                      "targetKey": "output", "targetIndex": 0},
                                             func="predictScore", input=algo_inputs))
            rank_task_name_2 = "algotransform_rank_%s" % model_info.name
            feature_config.add_algoTransform(name=rank_task_name_2,
                                             taskName="AlgoInference", feature=[rank_task_name_1],
                                             options={"algo-name": model_info.name,
                                                      "host": "${MODEL_HOST:localhost}",
                                                      "port": "${MODEL_PORT:50000}"},
                                             fieldActions=field_actions,
                                             output=[self.user_key, self.item_key, "score", "origin_scores"])
            recommend_config.add_service(name=service_name,
                                         preTransforms=[TransformConfig(name="summary")],
                                         columns=[{self.user_key: self.user_key_type},
                                                  {self.item_key: self.item_key_type},
                                                  {"score": "double"}, {"origin_scores": "map_str_double"}],
                                         tasks=[rank_task_name_2], options={"maxReservation": 200})
            service_dict[service_name] = service_name

    def gen_server_config(self):
        feature_config = FeatureConfig(source=[Source(name="request"), ])
        self.process_feature_source(feature_config)
        self.process_feature_sourceTable(feature_config)
        self.add_user_profile(feature_config)
        self.add_item_summary(feature_config)
        service_dict = dict()
        recommend_config = RecommendConfig()
        self.add_cf_recalls(service_dict, feature_config, recommend_config)
        self.add_rank_models(service_dict, feature_config, recommend_config)
        recommend_experiments = self.process_expriments(service_dict, recommend_config)
        layer_set = self.process_layers(recommend_experiments, recommend_config)
        self.process_scenes(layer_set, feature_config, recommend_config)
        online_configure = OnlineServiceConfig(feature_config, recommend_config)
        return DumpToYaml(online_configure).encode("utf-8").decode("latin1")
