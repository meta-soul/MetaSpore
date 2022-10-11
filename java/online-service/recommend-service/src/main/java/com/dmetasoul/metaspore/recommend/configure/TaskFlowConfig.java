//
// Copyright 2022 DMetaSoul
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
package com.dmetasoul.metaspore.recommend.configure;

import com.dmetasoul.metaspore.recommend.enums.DataTypeEnum;
import com.dmetasoul.metaspore.recommend.enums.JoinTypeEnum;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.google.common.collect.Sets;
import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.commons.collections4.CollectionUtils;
import org.apache.commons.collections4.MapUtils;
import org.apache.commons.lang3.StringUtils;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.cloud.context.config.annotation.RefreshScope;
import org.springframework.stereotype.Component;
import org.springframework.util.Assert;

import javax.annotation.PostConstruct;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * 服务配置汇总与初始化
 * Created by @author qinyy907 in 14:24 22/07/15.
 */
@Slf4j
@Data
@RefreshScope
@Component
public class TaskFlowConfig {
    @Autowired
    private FeatureConfig featureConfig;

    @Autowired
    private RecommendConfig recommendConfig;

    private Map<String, FeatureConfig.Source> sources = Maps.newHashMap();
    private Map<String, FeatureConfig.SourceTable> sourceTables = Maps.newHashMap();
    private Map<String, FeatureConfig.Feature> features = Maps.newHashMap();
    private Map<String, FeatureConfig.AlgoTransform> algoTransforms = Maps.newHashMap();
    private Map<String, RecommendConfig.Service> services = Maps.newHashMap();
    private Map<String, RecommendConfig.Experiment> experiments = Maps.newHashMap();
    private Map<String, RecommendConfig.Layer> layers = Maps.newHashMap();
    private Map<String, RecommendConfig.Scene> scenes = Maps.newHashMap();
    private Map<String, List<String>> featureRelyServices = Maps.newHashMap();

    @PostConstruct
    public void checkAndInit() {
        log.info("featureConfig: {}", featureConfig);
        log.info("recommendConfig: {}", recommendConfig);
        featureCheckAndInit();
        recommendCheckAndInit();
        checkFeatureAndInit();
        checkAlgoTransform();
    }

    public void featureCheckAndInit() {
        if (featureConfig == null) return;
        if (CollectionUtils.isNotEmpty(featureConfig.getSource())) {
            for (FeatureConfig.Source item : featureConfig.getSource()) {
                if (!item.checkAndDefault()) {
                    log.error("Source item {} is check fail!", item.getName());
                    throw new RuntimeException("Source check fail!");
                }
                sources.put(item.getName(), item);
            }
        }
        if (CollectionUtils.isNotEmpty(featureConfig.getSourceTable())) {
            for (FeatureConfig.SourceTable item : featureConfig.getSourceTable()) {
                FeatureConfig.Source source = sources.get(item.getSource());
                if (source == null) {
                    log.error("SourceTable: {} source {} is not config!", item.getName(), item.getSource());
                    throw new RuntimeException("SourceTable check fail!");
                }
                if (!item.checkAndDefault()) {
                    log.error("SourceTable item {} is check fail!", item.getName());
                    throw new RuntimeException("SourceTable check fail!");
                }
                if (source.getKind().equalsIgnoreCase("redis") && item.getColumnNames().size() != 2) {
                    log.error("SourceTable: {} from redis column size only support 2!", item.getName());
                    throw new RuntimeException("SourceTable check fail!");
                }
                item.setKind(source.getKind());
                if (item.getKind().equalsIgnoreCase("Redis") || item.getKind().equalsIgnoreCase("MongoDB")
                        || item.getKind().equalsIgnoreCase("JDBC")) {
                    item.setTaskName(item.getKind() + item.getTaskName());
                }
                sourceTables.put(item.getName(), item);
            }
        }
        if (CollectionUtils.isNotEmpty(featureConfig.getFeature())) {
            for (FeatureConfig.Feature item : featureConfig.getFeature()) {
                if (!item.checkAndDefault()) {
                    log.error("Feature item {} is check fail!", item.getName());
                    throw new RuntimeException("Feature check fail!");
                }
                features.put(item.getName(), item);
            }
        }
        if (CollectionUtils.isNotEmpty(featureConfig.getAlgoTransform())) {
            for (FeatureConfig.AlgoTransform item : featureConfig.getAlgoTransform()) {
                if (!item.checkAndDefault()) {
                    log.error("AlgoTransform item {} is check fail!", item.getName());
                    throw new RuntimeException("AlgoTransform check fail!");
                }
                algoTransforms.put(item.getName(), item);
            }
        }
        log.info("algoTransforms: {}", algoTransforms);
    }

    public void recommendCheckAndInit() {
        if (recommendConfig == null) return;
        if (CollectionUtils.isNotEmpty(recommendConfig.getServices())) {
            for (RecommendConfig.Service item : recommendConfig.getServices()) {
                if (!item.checkAndDefault()) {
                    log.error("Service item {} is check fail!", item.getName());
                    throw new RuntimeException("Service check fail!");
                }
                services.put(item.getName(), item);
            }
        }
        if (CollectionUtils.isNotEmpty(recommendConfig.getExperiments())) {
            for (RecommendConfig.Experiment item : recommendConfig.getExperiments()) {
                if (!item.checkAndDefault()) {
                    log.error("Experiment item {} is check fail!", item.getName());
                    throw new RuntimeException("Experiment check fail!");
                }
                experiments.put(item.getName(), item);
            }
        }
        if (CollectionUtils.isNotEmpty(recommendConfig.getLayers())) {
            for (RecommendConfig.Layer item : recommendConfig.getLayers()) {
                for (ExperimentItem experimentItem : item.getExperiments()) {
                    if (!experiments.containsKey(experimentItem.getName())) {
                        log.error("Layer: {} depend {} is not config!", item.getName(), experimentItem.getName());
                        throw new RuntimeException("Layer check fail!");
                    }
                }
                if (!item.checkAndDefault()) {
                    log.error("Feature item {} is check fail!", item.getName());
                    throw new RuntimeException("Layer check fail!");
                }
                layers.put(item.getName(), item);
            }
        }
        if (CollectionUtils.isNotEmpty(recommendConfig.getScenes())) {
            for (RecommendConfig.Scene item : recommendConfig.getScenes()) {
                if (!item.checkAndDefault()) {
                    log.error("AlgoTransform item {} is check fail!", item.getName());
                    throw new RuntimeException("Scene check fail!");
                }
                scenes.put(item.getName(), item);
            }
        }
    }

    private void checkAlgoTransform() {
        if (featureConfig == null) return;
        if (CollectionUtils.isEmpty(featureConfig.getAlgoTransform())) return;
        for (FeatureConfig.AlgoTransform item : featureConfig.getAlgoTransform()) {
            Map<String, String> fieldMap = Maps.newHashMap();
            Set<String> dependSet = Sets.newHashSet();
            //  检查依赖的feature是否存在
            if (CollectionUtils.isNotEmpty(item.getFeature())) {
                for (String featureItem : item.getFeature()) {
                    FeatureConfig.Feature feature = features.get(featureItem);
                    if (feature == null) {
                        log.error("AlgoTransform: {} Feature {} is not config!", item.getName(), featureItem);
                        throw new RuntimeException("AlgoTransform check fail!");
                    }
                    for (String col : feature.getColumnNames()) {
                        if (fieldMap.containsKey(col)) {
                            fieldMap.put(col, null);
                        } else {
                            fieldMap.put(col, featureItem);
                        }
                    }
                    dependSet.add(featureItem);
                }
            }
            //  检查依赖的algoTransform是否存在
            if (CollectionUtils.isNotEmpty(item.getAlgoTransform())) {
                for (String algoTransformItem : item.getAlgoTransform()) {
                    Assert.isTrue(!algoTransformItem.equals(item.getName()), "algotransform can not depend self! at " + algoTransformItem);
                    FeatureConfig.AlgoTransform algoTransform = algoTransforms.get(algoTransformItem);
                    if (algoTransform == null) {
                        log.error("AlgoTransform: {} depend algoTransform {} is not config!", item.getName(), algoTransformItem);
                        throw new RuntimeException("AlgoTransform check fail!");
                    }
                    for (String col : algoTransform.getColumnNames()) {
                        if (fieldMap.containsKey(col)) {
                            fieldMap.put(col, null);
                        } else {
                            fieldMap.put(col, algoTransformItem);
                        }
                    }
                    dependSet.add(algoTransformItem);
                }
            }
            //  补全涉及的field的table名称，以及检查table是否在依赖表中
            for (FieldAction fieldAction : item.getActionList()) {
                List<FieldInfo> fields = fieldAction.getFields();
                if (CollectionUtils.isNotEmpty(fields)) {
                    for (FieldInfo field : fields) {
                        if (StringUtils.isEmpty(field.getTable())) {
                            if (!fieldMap.containsKey(field.getFieldName()) || fieldMap.get(field.getFieldName()) == null) {
                                log.error("AlgoTransform: {} Field {} not exist!", item.getName(), field);
                                throw new RuntimeException("AlgoTransform check fail!");
                            }
                            field.setTable(fieldMap.get(field.getFieldName()));
                        } else if (!dependSet.contains(field.getTable())) {
                            log.error("AlgoTransform {} fieldAction fields {} table must in depen!", item.getName(), field);
                            throw new RuntimeException("AlgoTransform check fail!");
                        }
                    }
                }
            }
        }
    }

    private void checkFeatureAndInit() {
        if (featureConfig == null) return;
        if (CollectionUtils.isEmpty(featureConfig.getFeature())) return;
        for (FeatureConfig.Feature item : featureConfig.getFeature()) {
            // 记录不需要生成查询条件即可查询数据的表， 如request， 以及除sourceTable外计算的结果表
            Set<String> immediateSet = Sets.newHashSet();
            // 用于补全field 的table, key: 字段名， value: 表名， 出现多个相同字段名则value=null
            Map<String, String> fieldMap = Maps.newHashMap();
            Map<String, Map<String, Field>> fieldTypes = Maps.newHashMap();
            Map<String, Map<String, DataTypeEnum>> columnTypes = Maps.newHashMap();
            // 记录每个from表里的字段列表，用于后续计算
            Map<String, List<String>> fromColumns = Maps.newHashMap();
            // 检查from里的数据表是否存在
            for (String rely : item.getFrom()) {
                if (!sourceTables.containsKey(rely) && !features.containsKey(rely) && !services.containsKey(rely) && !algoTransforms.containsKey(rely)) {
                    log.error("Feature: {} rely {} is not config!", item.getName(), rely);
                    throw new RuntimeException("Feature check fail!");
                }
                // 默认feature， chain数据都是计算好的中间结果，可以直接获取到
                if (!sourceTables.containsKey(rely) || sourceTables.get(rely).getKind().equalsIgnoreCase("request")) {
                    immediateSet.add(rely);
                }
                List<String> columnNames = null;
                if (features.containsKey(rely)) {
                    FeatureConfig.Feature feature = features.get(rely);
                    columnNames = feature.getColumnNames();
                    columnTypes.put(rely, feature.getColumnMap());
                    fieldTypes.put(rely, feature.getFieldMap());
                }
                if (sourceTables.containsKey(rely)) {
                    FeatureConfig.SourceTable sourceTable = sourceTables.get(rely);
                    columnNames = sourceTable.getColumnNames();
                    columnTypes.put(rely, sourceTable.getColumnMap());
                    fieldTypes.put(rely, sourceTable.getFieldMap());
                }
                if (services.containsKey(rely)) {
                    RecommendConfig.Service service = services.get(rely);
                    columnNames = service.getColumnNames();
                    Assert.notNull(columnNames, "the columns info must configure in " + rely);
                    columnTypes.put(rely, service.getColumnMap());
                    fieldTypes.put(rely, service.getFieldMap());
                    featureRelyServices.computeIfAbsent(item.getName(), key -> Lists.newArrayList()).add(rely);
                }
                if (algoTransforms.containsKey(rely)) {
                    FeatureConfig.AlgoTransform algoTransform = algoTransforms.get(rely);
                    columnNames = algoTransform.getColumnNames();
                    columnTypes.put(rely, algoTransform.getColumnMap());
                    fieldTypes.put(rely, algoTransform.getFieldMap());
                }
                fromColumns.put(rely, columnNames);
                if (CollectionUtils.isNotEmpty(columnNames)) {
                    for (String col : columnNames) {
                        if (fieldMap.containsKey(col)) {
                            fieldMap.put(col, null);
                        } else {
                            fieldMap.put(col, rely);
                        }
                    }
                }
            }
            Map<String, DataTypeEnum> columns = Maps.newHashMap();
            Map<String, Field> fields = Maps.newHashMap();
            for (int index = 0; index < item.getFields().size(); ++index) {
                FieldInfo field = item.getFields().get(index);
                if (field.getTable() == null) {
                    if (fieldMap.get(field.getFieldName()) == null) {
                        log.error("Feature: {} select field: {} has wrong column field!", item.getName(), field.fieldName);
                        throw new RuntimeException("Feature check fail!");
                    }
                    field.setTable(fieldMap.get(field.getFieldName()));
                }
                Map<String, DataTypeEnum> columnType = columnTypes.get(field.getTable());
                columns.put(field.getFieldName(), columnType.get(field.getFieldName()));
                fields.put(field.getFieldName(), fieldTypes.get(field.getTable()).get(field.getFieldName()));
            }
            if (MapUtils.isNotEmpty(item.getFilterMap())) {
                for (Map.Entry<FieldInfo, Map<FieldInfo, String>> entry : item.getFilterMap().entrySet()) {
                    FieldInfo field = entry.getKey();
                    if (field.getTable() != null && !columnTypes.containsKey(field.getTable())) {
                        log.error("Feature config the field of filters, field table' must be in the rely!");
                        throw new RuntimeException("Feature check fail!");
                    }
                    if (field.getTable() == null) {
                        if (fieldMap.get(field.getFieldName()) == null) {
                            log.error("Feature: {} filters field: {} has wrong column field!", item.getName(), field.fieldName);
                            throw new RuntimeException("Feature check fail!");
                        }
                        field.setTable(fieldMap.get(field.getFieldName()));
                    }
                    for (Map.Entry<FieldInfo, String> entry1 : entry.getValue().entrySet()) {
                        FieldInfo field1 = entry1.getKey();
                        if (field1.getTable() != null && !columnTypes.containsKey(field1.getTable())) {
                            log.error("Feature config the field of filter, field table' must be in the rely!");
                            throw new RuntimeException("Feature check fail!");
                        }
                        if (field1.getTable() == null) {
                            if (fieldMap.get(field1.getFieldName()) == null) {
                                log.error("Feature: {} filter field: {} has wrong column field!", item.getName(), field1.fieldName);
                                throw new RuntimeException("Feature check fail!");
                            }
                            field1.setTable(fieldMap.get(field1.getFieldName()));
                        }
                    }
                }
            }
            item.setColumnMap(columns);
            item.setFieldMap(fields);
            item.setFromColumns(fromColumns);
            Map<String, List<Condition>> conditionMap = Maps.newHashMap();
            if (CollectionUtils.isNotEmpty(item.getCondition())) {
                for (int index = 0; index < item.getCondition().size(); ++index) {
                    Condition cond = item.getCondition().get(index);
                    if (cond.left.getTable().equals(cond.right.getTable())) {
                        log.error("Feature: {} condition table {} field: {} has join self field:{}!",
                                item.getName(), cond.left.getTable(), cond.left.getFieldName(), cond.right.getFieldName());
                        throw new RuntimeException("Feature check fail!");
                    }
                    if (cond.getType() == JoinTypeEnum.INNER) {
                        conditionMap.computeIfAbsent(cond.left.getTable(), key -> Lists.newArrayList()).add(cond);
                        conditionMap.computeIfAbsent(cond.right.getTable(), key -> Lists.newArrayList()).add(Condition.reverse(cond));
                    } else if (cond.getType() == JoinTypeEnum.LEFT) {
                        conditionMap.computeIfAbsent(cond.right.getTable(), key -> Lists.newArrayList()).add(Condition.reverse(cond));
                    } else if (cond.getType() == JoinTypeEnum.RIGHT) {
                        conditionMap.computeIfAbsent(cond.left.getTable(), key -> Lists.newArrayList()).add(cond);
                    }
                }
            }
            item.getFrom().forEach(x -> {
                if (!immediateSet.contains(x) && !conditionMap.containsKey(x)) {
                    immediateSet.add(x);
                }
            });
            if (immediateSet.isEmpty()) {
                immediateSet.add(item.getFrom().get(0));
            }
            // immediateList 的意义在于优先计算没有join依赖条件的数据，sourceTable数据获取需要请求条件
            List<String> immediateList = Lists.newArrayList();
            immediateList.addAll(immediateSet);
            item.setImmediateFrom(immediateList);
            item.setConditionMap(conditionMap);
        }
    }
}
