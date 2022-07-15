package com.dmetasoul.metaspore.recommend.configure;

import com.dmetasoul.metaspore.recommend.enums.JoinTypeEnum;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.google.common.collect.Sets;
import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.collections4.CollectionUtils;
import org.apache.commons.collections4.MapUtils;
import org.apache.commons.lang3.StringUtils;
import org.apache.commons.lang3.tuple.Pair;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.cloud.context.config.annotation.RefreshScope;
import org.springframework.stereotype.Component;

import javax.annotation.PostConstruct;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

@Slf4j
@Data
@RefreshScope
@Component
public class TaskFlowConfig {
    @Autowired
    private FeatureConfig featureConfig;

    @Autowired
    private RecommendConfig recommendConfig;

    @Autowired
    private FunctionConfig functionConfig;

    private Map<String, FeatureConfig.Source> sources = Maps.newHashMap();
    private Map<String, FeatureConfig.SourceTable> sourceTables = Maps.newHashMap();
    private Map<String, FeatureConfig.Feature> features = Maps.newHashMap();
    private Map<String, FeatureConfig.AlgoTransform> algoTransforms = Maps.newHashMap();

    private Map<String, RecommendConfig.Chain> chains = Maps.newHashMap();
    private Map<String, RecommendConfig.Service> services = Maps.newHashMap();
    private Map<String, RecommendConfig.Experiment> experiments = Maps.newHashMap();
    private Map<String, RecommendConfig.Layer> layers = Maps.newHashMap();
    private Map<String, RecommendConfig.Scene> scenes = Maps.newHashMap();

    @PostConstruct
    public void checkAndInit() {
        for (FeatureConfig.Source item: featureConfig.getSource()) {
            if (!item.checkAndDefault()) {
                log.error("Source item {} is check fail!", item.getName());
                throw new RuntimeException("Source check fail!");
            }
            sources.put(item.getName(), item);
        }
        for (FeatureConfig.SourceTable item: featureConfig.getSourceTable()) {
            FeatureConfig.Source source = sources.get(item.getSource());
            if (source == null) {
                log.error("SourceTable: {} source {} is not config!", item.getName(), item.getSource());
                throw new RuntimeException("SourceTable check fail!");
            }
            if (!item.checkAndDefault()) {
                log.error("SourceTable item {} is check fail!", item.getName());
                throw new RuntimeException("SourceTable check fail!");
            }
            if (source.getKind().equals("redis") && item.getColumnNames().size() != 2) {
                log.error("SourceTable: {} from redis column size only support 2!", item.getName());
                throw new RuntimeException("SourceTable check fail!");
            }
            sourceTables.put(item.getName(), item);
        }
        for (FeatureConfig.Feature item: featureConfig.getFeature()) {
            if (!item.checkAndDefault()) {
                log.error("Feature item {} is check fail!", item.getName());
                throw new RuntimeException("Feature check fail!");
            }
            features.put(item.getName(), item);
        }
        for (FeatureConfig.AlgoTransform item : featureConfig.getAlgoTransform()) {
            FeatureConfig.Feature feature = features.get(item.getDepend());
            if (feature == null) {
                log.error("AlgoTransform: {} Feature {} is not config!", item.getName(), item.getDepend());
                throw new RuntimeException("AlgoTransform check fail!");
            }
            if (!item.checkAndDefault()) {
                log.error("AlgoTransform item {} is check fail!", item.getName());
                throw new RuntimeException("AlgoTransform check fail!");
            }
            algoTransforms.put(item.getName(), item);
        }
        for (RecommendConfig.Service item: recommendConfig.getServices()) {
            if (!item.checkAndDefault()) {
                log.error("Service item {} is check fail!", item.getName());
                throw new RuntimeException("Service check fail!");
            }
            services.put(item.getName(), item);
        }
        for (RecommendConfig.Experiment item: recommendConfig.getExperiments()) {
            if (!item.checkAndDefault()) {
                log.error("Experiment item {} is check fail!", item.getName());
                throw new RuntimeException("Experiment check fail!");
            }
            List<String> columns = null;
            int chainNum = item.getChains().size();
            for (int index = 0; index < chainNum; ++index) {
                RecommendConfig.Chain chain = item.getChains().get(index);
                if (StringUtils.isNotEmpty(chain.getName())) {
                    chains.put(chain.getName(), chain);
                }
                if (CollectionUtils.isNotEmpty(chain.getThen())) {
                    for (int i = 0; i < chain.getThen().size(); ++i) {
                        String rely = chain.getThen().get(i);
                        if (!services.containsKey(rely)) {
                            log.error("Experiment: {} Service {} is not config in then!", item.getName(), rely);
                            throw new RuntimeException("Experiment check fail!");
                        }
                        if (CollectionUtils.isEmpty(chain.getColumnNames()) && i == chain.getThen().size() - 1) {
                            RecommendConfig.Service service = services.get(rely);
                            chain.setColumnMap(service.getColumns());
                        }
                    }
                }
                if (CollectionUtils.isNotEmpty(chain.getWhen())) {
                    for (String rely : chain.getWhen()) {
                        if (!services.containsKey(rely)) {
                            log.error("Experiment: {} Service {} is not config in when!", item.getName(), rely);
                            throw new RuntimeException("Experiment check fail!");
                        }
                    }
                }
            }

            if (chainNum > 0) {
                item.setColumnMap(item.getChains().get(chainNum-1).getColumns());
            }
            if (MapUtils.isNotEmpty(item.getOptions()) && item.getOptions().containsKey("cutField")) {
                String cutField = (String) item.getOptions().get("cutField");
                if (!item.getColumnMap().containsKey(cutField)) {
                    log.error("Experiment: {} cutField must in output columns!", item.getName());
                    throw new RuntimeException("Experiment check fail!");
                }
            }
            experiments.put(item.getName(), item);
        }
        for (RecommendConfig.Layer item: recommendConfig.getLayers()) {
            for (RecommendConfig.ExperimentItem experimentItem : item.getExperiments()) {
                if (!experiments.containsKey(experimentItem.getName())) {
                    log.error("Layer: {} depend {} is not config!", item.getName(), experimentItem.getName());
                    throw new RuntimeException("Layer check fail!");
                }
            }
            if (!item.checkAndDefault()) {
                log.error("Feature item {} is check fail!", item.getName());
                throw new RuntimeException("Layer check fail!");
            }
            RecommendConfig.Experiment experiment = experiments.get(item.getExperiments().get(0).getName());
            item.setColumnMap(experiment.getColumns());
            layers.put(item.getName(), item);
        }
        for (RecommendConfig.Scene item: recommendConfig.getScenes()) {
            if (!item.checkAndDefault()) {
                log.error("AlgoTransform item {} is check fail!", item.getName());
                throw new RuntimeException("Scene check fail!");
            }
            for (RecommendConfig.Chain chain : item.getChains()) {
                if (!chain.checkAndDefault()) {
                    log.error("Scene {} chain {} is check fail!", item.getName(), chain.getName());
                    throw new RuntimeException("Scene check fail!");
                }
                if (StringUtils.isNotEmpty(chain.getName())) {
                    chains.put(chain.getName(), chain);
                }
                if (CollectionUtils.isNotEmpty(chain.getThen())) {
                    for (int i = 0; i < chain.getThen().size(); ++i) {
                        String rely = chain.getThen().get(i);
                        RecommendConfig.Layer layer = layers.get(rely);
                        if (layer == null) {
                            log.error("Scene: {} Layer {} is not config in then!", item.getName(), rely);
                            throw new RuntimeException("Scene check fail!");
                        }
                        if (CollectionUtils.isEmpty(chain.getColumnNames()) && i == chain.getThen().size() - 1) {
                            item.setColumnMap(layer.getColumns());
                        }
                    }
                }
                if (CollectionUtils.isNotEmpty(chain.getWhen())) {
                    for (String rely : chain.getWhen()) {
                        if (!layers.containsKey(rely)) {
                            log.error("Scene: {} Layer {} is not config in when!", item.getName(), rely);
                            throw new RuntimeException("Scene check fail!");
                        }
                    }
                }
            }
            scenes.put(item.getName(), item);
        }
        checkFeatureAndInit();
        checkAlgoTransform();
        if (!functionConfig.checkAndInit()) {
            log.error("Function init fail!");
            throw new RuntimeException("Function checkAndInit fail!");
        }
    }

    private void checkAlgoTransform() {
        for (FeatureConfig.AlgoTransform item : featureConfig.getAlgoTransform()) {
            FeatureConfig.Feature feature = features.get(item.getDepend());
            if (feature == null) {
                log.error("AlgoTransform: {} Feature {} is not config!", item.getName(), item.getDepend());
                throw new RuntimeException("AlgoTransform check fail!");
            }
            for (int index = 0; index < item.getFieldActions().size(); ++index) {
                FeatureConfig.FieldAction fieldAction = item.getFieldActions().get(index);
                List<String> fields = fieldAction.getFields();
                for (String field : fields) {
                    if (!feature.getColumnMap().containsKey(field)) {
                        log.error("AlgoTransform {} fieldAction fields {} must in feature columns!", item.getName(), field);
                        throw new RuntimeException("AlgoTransform check fail!");
                    }
                }
                if (fields.size() == 1 && StringUtils.isEmpty(fieldAction.getType())) {
                    fieldAction.setType(feature.getColumnMap().get(fields.get(0)));
                }
            }
        }
    }

    private void checkFeatureAndInit() {
        for (FeatureConfig.Feature item: featureConfig.getFeature()) {
            Set<String> immediateSet = Sets.newHashSet();
            if (CollectionUtils.isNotEmpty(item.getImmediateFrom())) {
                immediateSet.addAll(item.getImmediateFrom());
            }
            Map<String, String> fieldMap = Maps.newHashMap();
            Map<String, Map<String, String>> columnTypes = Maps.newHashMap();
            Map<String, List<String>> fromColumns = Maps.newHashMap();
            for (String rely : item.getFrom()) {
                if (!sourceTables.containsKey(rely) && !features.containsKey(rely) && !chains.containsKey(rely)) {
                    log.error("Feature: {} rely {} is not config!", item.getName(), rely);
                    throw new RuntimeException("Feature check fail!");
                }
                // 默认feature， chain数据都是计算好的中间结果，可以直接获取到
                if (features.containsKey(rely) || chains.containsKey(rely) || sources.get(sourceTables.get(rely).getSource()).getKind().equals("request")) {
                    immediateSet.add(rely);
                }
                List<String> columnNames = null;
                if (features.containsKey(rely)) {
                    columnNames = features.get(rely).getColumnNames();
                    columnTypes.put(rely, features.get(rely).getColumnMap());
                }
                if (sourceTables.containsKey(rely)) {
                    columnNames = sourceTables.get(rely).getColumnNames();
                    columnTypes.put(rely, sourceTables.get(rely).getColumnMap());
                }
                if (chains.containsKey(rely)) {
                    columnNames = chains.get(rely).getColumnNames();
                    columnTypes.put(rely, chains.get(rely).getColumnMap());
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
            Map<String, String> columns = Maps.newHashMap();
            for (int index = 0; index < item.getFields().size(); ++index) {
                FeatureConfig.Feature.Field field = item.getFields().get(index);
                if (field.getTable() == null) {
                    if (fieldMap.get(field.getFieldName()) == null) {
                        log.error("Feature: {} select field: {} has wrong column field!", item.getName(), field.fieldName);
                        throw new RuntimeException("Feature check fail!");
                    }
                    field.setTable(fieldMap.get(field.getFieldName()));
                }
                Map<String, String> columnType = columnTypes.get(field.getTable());
                columns.put(field.getFieldName(), columnType.get(field.getFieldName()));
            }
            item.setColumnMap(columns);
            item.setFromColumns(fromColumns);
            Map<String, List<FeatureConfig.Feature.Condition>> conditionMap = Maps.newHashMap();
            if (CollectionUtils.isNotEmpty(item.getCondition())) {
                for (int index = 0; index < item.getCondition().size(); ++index) {
                    FeatureConfig.Feature.Condition cond = item.getCondition().get(index);
                    if (cond.left.getTable().equals(cond.right.getTable())) {
                        log.error("Feature: {} condition table {} field: {} has join self field:{}!",
                                item.getName(), cond.left.getTable(), cond.left.getFieldName(), cond.right.getFieldName());
                        throw new RuntimeException("Feature check fail!");
                    }
                    conditionMap.putIfAbsent(cond.left.getTable(), Lists.newArrayList());
                    conditionMap.putIfAbsent(cond.right.getTable(), Lists.newArrayList());
                    conditionMap.get(cond.left.getTable()).add(cond);
                    conditionMap.get(cond.right.getTable()).add(cond);
                }
            }
            item.getFrom().forEach(x->{
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
