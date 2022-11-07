package com.dmetasoul.metaspore.baseservice;

import com.dmetasoul.metaspore.common.CommonUtils;
import com.dmetasoul.metaspore.common.Utils;
import com.dmetasoul.metaspore.configure.AlgoTransform;
import com.dmetasoul.metaspore.configure.RecommendConfig;
import com.dmetasoul.metaspore.configure.TaskFlowConfig;
import com.dmetasoul.metaspore.data.DataContext;
import com.dmetasoul.metaspore.data.DataResult;
import com.dmetasoul.metaspore.data.ServiceRequest;
import com.dmetasoul.metaspore.data.ServiceResult;
import com.dmetasoul.metaspore.dataservice.DataService;
import com.dmetasoul.metaspore.enums.DataTypeEnum;
import com.dmetasoul.metaspore.recommend.Experiment;
import com.dmetasoul.metaspore.recommend.Layer;
import com.dmetasoul.metaspore.recommend.Scene;
import com.dmetasoul.metaspore.recommend.Service;
import com.google.common.collect.Lists;
import lombok.Data;
import lombok.SneakyThrows;
import lombok.extern.slf4j.Slf4j;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.commons.collections4.CollectionUtils;
import org.apache.commons.lang3.StringUtils;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;
import org.springframework.util.StopWatch;

import java.util.List;
import java.util.Map;
import java.util.UUID;


@Slf4j
@Data
@Component
public class RecommendService {
    @Autowired
    public TaskFlowConfig taskFlowConfig;

    @Autowired
    public TaskServiceRegister taskServiceRegister;

    @SuppressWarnings("unchecked")
    public ServiceResult getDataServiceResult(String task, Map<String, Object> req) {
        DataService taskService = taskServiceRegister.getDataService(task);
        if (taskService == null) {
            return ServiceResult.of(-1, "taskService is not exist!");
        }
        try (DataContext context = new DataContext(req)) {
            List<String> services = null;
            if (taskFlowConfig.getFeatures().containsKey(task) && taskFlowConfig.getFeatureRelyServices().containsKey(task)) {
                services = taskFlowConfig.getFeatureRelyServices().get(task);
            }
            if (taskFlowConfig.getAlgoTransforms().containsKey(task)) {
                AlgoTransform algoTransform = taskFlowConfig.getAlgoTransforms().get(task);
                services = getRelyServiceList(algoTransform);
            }
            if (CollectionUtils.isNotEmpty(services)) {
                for (String item : services) {
                    if (!req.containsKey(item) || !(req.get(item) instanceof List)) {
                        return ServiceResult.of(-1, "taskService need depend data: " + item);
                    }
                    List<Map<String, Object>> data = (List<Map<String, Object>>) req.get(item);
                    RecommendConfig.Service serviceConfig = taskFlowConfig.getServices().get(item);
                    List<Field> fields = Lists.newArrayList();
                    List<DataTypeEnum> types = Lists.newArrayList();
                    if (CollectionUtils.isNotEmpty(serviceConfig.getColumnNames())) {
                        for (String col : serviceConfig.getColumnNames()) {
                            fields.add(serviceConfig.getFieldMap().get(col));
                            types.add(serviceConfig.getColumnMap().get(col));
                        }
                        DataResult resultItem = new DataResult();
                        resultItem.setFeatureData(item, fields, types, data);
                        taskService.setDataResultByName(item, resultItem, context);
                    }
                }
            }
            StopWatch timeRecorder = new StopWatch(UUID.randomUUID().toString());
            timeRecorder.start(String.format("task_%s_total", task));
            try (DataResult result = taskService.execute(new ServiceRequest(req), context)) {
                timeRecorder.stop();
                context.updateTimeRecords(Utils.getTimeRecords(timeRecorder));
                if (result == null) {
                    return ServiceResult.of(-1, "taskService execute fail!");
                }
                return ServiceResult.of(result.output()).addTimeRecord(context.getTimeRecords());
            }
        }
    }

    private List<String> getRelyServiceList(AlgoTransform algoTransform) {
        List<String> services = Lists.newArrayList();
        if (CollectionUtils.isNotEmpty(algoTransform.getFeature())) {
            for (String table : algoTransform.getFeature()) {
                if (taskFlowConfig.getFeatureRelyServices().containsKey(table)) {
                    services.addAll(taskFlowConfig.getFeatureRelyServices().get(table));
                }
            }
        }
        if (CollectionUtils.isNotEmpty(algoTransform.getAlgoTransform())) {
            for (String table : algoTransform.getAlgoTransform()) {
                AlgoTransform algo = taskFlowConfig.getAlgoTransforms().get(table);
                services.addAll(getRelyServiceList(algo));
            }
        }
        return services;
    }

    @SneakyThrows
    private List<DataResult> executeTasks(List<DataResult> input, List<String> tasks, DataContext context) {
        List<DataResult> result = Lists.newArrayList();
        if (CollectionUtils.isNotEmpty(tasks)) {
            for (String task : tasks) {
                if (taskServiceRegister.getRecommendServices().containsKey(task)) {
                    Service taskService = taskServiceRegister.getRecommendService(task);
                    result.addAll(taskService.execute(input, context).get());
                } else if (taskServiceRegister.getExperimentMap().containsKey(task)) {
                    Experiment taskService = taskServiceRegister.getExperiment(task);
                    result.addAll(taskService.process(input, context).get());
                } else if (taskServiceRegister.getLayerMap().containsKey(task)) {
                    Layer taskService = taskServiceRegister.getLayer(task);
                    result.addAll(taskService.execute(input, context).get());
                } else if (taskServiceRegister.getSceneMap().containsKey(task)) {
                    Scene taskService = taskServiceRegister.getScene(task);
                    result.add(taskService.process(context));
                }
            }
        }
        return result;
    }


    public ServiceResult getRecommendResult(String task, Map<String, Object> req) {
        try (DataContext context = new DataContext(req)) {
            StopWatch timeRecorder = new StopWatch(UUID.randomUUID().toString());
            timeRecorder.start(String.format("task_%s_total", task));
            List<String> preTasks = CommonUtils.getField(req, "preTasks", List.of());
            List<DataResult> result = executeTasks(executeTasks(List.of(), preTasks, context), List.of(task), context);
            log.info("recommend result : {}", result);
            if (CollectionUtils.isEmpty(result)) {
                timeRecorder.stop();
                return ServiceResult.of(-1, "taskService execute fail!");
            }
            List<Map<String, Object>> output = Lists.newArrayList();
            for (DataResult item : result) {
                output.addAll(item.output());
                item.close();
            }
            timeRecorder.stop();
            context.updateTimeRecords(Utils.getTimeRecords(timeRecorder));
            return ServiceResult.of(output).addTimeRecord(context.getTimeRecords());
        }
    }

    public ServiceResult recommend(String scene, String id, Map<String, Object> req) {
        Scene sceneService = taskServiceRegister.getScene(scene);
        if (sceneService == null) {
            return ServiceResult.of(-1, String.format("scene:%s is not support!", scene));
        }
        try (DataContext context = new DataContext(req)) {
            if (StringUtils.isEmpty(id)) {
                return ServiceResult.of(-1, String.format("scene:%s recommend need id, eg:userId!", scene));
            }
            context.setId(id);
            StopWatch timeRecorder = new StopWatch(UUID.randomUUID().toString());
            timeRecorder.start(String.format("scene_%s_total", scene));
            List<Map<String, Object>> data = sceneService.output(context);
            timeRecorder.stop();
            context.updateTimeRecords(Utils.getTimeRecords(timeRecorder));
            return ServiceResult.of(data, id).addTimeRecord(context.getTimeRecords());
        }
    }

    public ServiceResult itemSummary(String item_key, String id, Map<String, Object> req) {
        DataService taskService = taskServiceRegister.getDataService("feature_item_summary");
        if (taskService == null) {
            return ServiceResult.of(-1, "itemSummary is not support in configure!");
        }
        if (StringUtils.isEmpty(item_key)) {
            item_key = "item_id";
        }
        if (StringUtils.isEmpty(id)) {
            return ServiceResult.of(-1, "itemSummary need itemId!");
        }
        req.put(item_key, id);
        DataContext context = new DataContext(req);
        DataResult result;
        StopWatch timeRecorder = new StopWatch(UUID.randomUUID().toString());
        timeRecorder.start("itemSummary_total");
        result = taskService.execute(new ServiceRequest(req), context);
        timeRecorder.stop();
        context.updateTimeRecords(Utils.getTimeRecords(timeRecorder));
        if (result == null) {
            return ServiceResult.of(-1, "itemSummary execute fail!");
        }
        return ServiceResult.of(result.output()).addTimeRecord(context.getTimeRecords());
    }
}
