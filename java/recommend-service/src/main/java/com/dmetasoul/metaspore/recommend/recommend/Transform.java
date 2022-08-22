package com.dmetasoul.metaspore.recommend.recommend;

import com.dmetasoul.metaspore.recommend.common.Utils;
import com.dmetasoul.metaspore.recommend.configure.TransformConfig;
import com.dmetasoul.metaspore.recommend.data.DataContext;
import com.dmetasoul.metaspore.recommend.data.DataResult;
import com.dmetasoul.metaspore.recommend.enums.DataTypeEnum;
import com.dmetasoul.metaspore.recommend.recommend.interfaces.MergeOperator;
import com.dmetasoul.metaspore.recommend.recommend.interfaces.TransformFunction;
import com.dmetasoul.metaspore.serving.ArrowAllocator;
import com.dmetasoul.metaspore.serving.FeatureTable;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import lombok.extern.slf4j.Slf4j;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.commons.collections4.CollectionUtils;
import org.apache.commons.collections4.MapUtils;
import org.springframework.util.Assert;

import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;

@SuppressWarnings("unchecked")
@Slf4j
public abstract class Transform {
    private String name;
    private ExecutorService taskPool;
    protected Map<String, TransformFunction> transformFunctions;
    protected Map<String, MergeOperator> mergeOperators;

    protected List<Field> resFields;
    protected List<DataTypeEnum> dataTypes;

    public static final int DEFAULT_MAX_RESERVATION = 50;

    public void initTransform(String name, ExecutorService taskPool) {
        this.name = name;
        this.taskPool = taskPool;
        this.transformFunctions = Maps.newHashMap();
        this.mergeOperators = Maps.newHashMap();
        initFunctions();
        addFunctions();
    }

    public void addFunctions() {
        addFunction("summary", (data, results, context, option) -> {
            Assert.isTrue(CollectionUtils.isNotEmpty(resFields), "summary need configure columns info!");
            DataResult result = new DataResult();
            FeatureTable featureTable = new FeatureTable(name, resFields, ArrowAllocator.getAllocator());
            result.setFeatureTable(featureTable);
            result.setDataTypes(dataTypes);
            result.setName(name);
            List<String> dupFields = getOptionFields("dupFields", option);
            result.mergeDataResult(data, dupFields, getMergeOperators(option), option);
            featureTable.finish();
            results.add(result);
            return true;
        });
        addFunction("summaryBySchema", (data, results, context, option) -> {
            if (CollectionUtils.isNotEmpty(data)) {
                DataResult item = data.get(0);
                DataResult result = new DataResult();
                FeatureTable featureTable = new FeatureTable(String.format("%s.summaryBySchema", name), item.getFields(), ArrowAllocator.getAllocator());
                result.setFeatureTable(featureTable);
                result.setDataTypes(item.getDataTypes());
                List<String> dupFields = getOptionFields("dupFields", option);
                result.mergeDataResult(data, dupFields, getMergeOperators(option), option);
                featureTable.finish();
                results.add(result);
            }
            return true;
        });
        addFunction("orderAndLimit", (data, results, context, option) -> {
            if (CollectionUtils.isNotEmpty(data)) {
                for (DataResult item : data) {
                    DataResult result = new DataResult();
                    FeatureTable featureTable = new FeatureTable(item.getFeatureTable().getName(), item.getFields(), ArrowAllocator.getAllocator());
                    result.setFeatureTable(featureTable);
                    result.setDataTypes(item.getDataTypes());
                    List<String> orderFields = getOptionFields("orderFields", option);
                    int limit = Utils.getField(option, "maxReservation", DEFAULT_MAX_RESERVATION);
                    result.orderAndLimit(item, orderFields, limit);
                    featureTable.finish();
                    results.add(result);
                }
            }
            return true;
        });
        addFunction("cutOff", (data, results, context, option) -> {
            if (CollectionUtils.isNotEmpty(data)) {
                for (DataResult item : data) {
                    DataResult result = new DataResult();
                    FeatureTable featureTable = new FeatureTable(item.getFeatureTable().getName(), item.getFields(), ArrowAllocator.getAllocator());
                    result.setFeatureTable(featureTable);
                    result.setDataTypes(item.getDataTypes());
                    int limit = Utils.getField(option, "maxReservation", DEFAULT_MAX_RESERVATION);
                    result.copyDataResult(item, 0, limit);
                    featureTable.finish();
                    results.add(result);
                }
            }
            return true;
        });
    }

    public abstract void initFunctions();

    public void addFunction(String name, TransformFunction function) {
        transformFunctions.put(name, function);
    }

    public void registerOperator(String name, MergeOperator operator) {
        mergeOperators.put(name, operator);
    }

    private List<String> getOptionFields(String name, Map<String, Object> option) {
        List<String> dupFields = Lists.newArrayList();
        if (MapUtils.isNotEmpty(option)) {
            Object dupFieldValue = option.get(name);
            if (dupFieldValue == null) return dupFields;
            if (dupFieldValue instanceof Map) {
                dupFields.addAll(((Map) dupFieldValue).values());
            } else if (dupFieldValue instanceof Collection) {
                dupFields.addAll((Collection<String>) dupFieldValue);
            }
        }
        return dupFields;
    }

    public Map<String, MergeOperator> getMergeOperators(Map<String, Object> option) {
        Map<String, MergeOperator> mergeOperatorMap = Maps.newHashMap();
        Map<String, String> mergeFieldOperator = Utils.getField(option, "mergeOperator", Map.of());
        if (MapUtils.isNotEmpty(mergeFieldOperator)) {
            for (Map.Entry<String, String> entry : mergeFieldOperator.entrySet()) {
                MergeOperator operator = mergeOperators.get(entry.getValue());
                if (operator == null) {
                    log.error("merge operator no found config fail col: {}, operator: {}", entry.getKey(), entry.getValue());
                    continue;
                }
                mergeOperatorMap.put(entry.getKey(), operator);
            }
        }
        return mergeOperatorMap;
    }

    public CompletableFuture<List<DataResult>> executeTransform(CompletableFuture<List<DataResult>> future,
                                                                List<TransformConfig> transforms,
                                                                Map<String, Object> args,
                                                                DataContext context) {
        if (future == null || CollectionUtils.isEmpty(transforms)) return null;
        for (TransformConfig item : transforms) {
            TransformFunction function = transformFunctions.get(item.getName());
            if (function == null) {
                log.error("the service：{} function: {} is not exist!", name, item.getName());
                continue;
            }
            Map<String, Object> option = Maps.newHashMap();
            if (MapUtils.isNotEmpty(args)) {
                option.putAll(args);
            }
            if (MapUtils.isNotEmpty(item.getOption())) {
                option.putAll(item.getOption());
            }
            future = future.thenApplyAsync(dataResults -> {
                List<DataResult> resultList = Lists.newArrayList();
                if (CollectionUtils.isEmpty(dataResults)) {
                    log.error("the service：{} function: {} input is empty!", name, item.getName());
                    return resultList;
                }
                if (!function.transform(dataResults, resultList, context, option)) {
                    log.error("the service：{} function: {} execute fail!", name, item.getName());
                }
                return resultList;
            }, taskPool);
        }
        return future;
    }

}
