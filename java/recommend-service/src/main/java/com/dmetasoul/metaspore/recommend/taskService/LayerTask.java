package com.dmetasoul.metaspore.recommend.dataservice;

import com.dmetasoul.metaspore.recommend.annotation.DataServiceAnnotation;
import com.dmetasoul.metaspore.recommend.configure.RecommendConfig;
import com.dmetasoul.metaspore.recommend.data.DataContext;
import com.dmetasoul.metaspore.recommend.data.DataResult;
import com.dmetasoul.metaspore.recommend.data.ServiceRequest;
import com.dmetasoul.metaspore.recommend.enums.TaskStatusEnum;
import com.dmetasoul.metaspore.recommend.bucketizer.LayerBucketizer;
import com.google.common.collect.Maps;
import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.collections4.MapUtils;
import org.springframework.stereotype.Component;

import java.util.List;
import java.util.Map;

@SuppressWarnings("rawtypes")
@Data
@Slf4j
@DataServiceAnnotation("Layer")
public class LayerTask extends DataService {

    private RecommendConfig.Layer layer;
    private LayerBucketizer bucketizer;

    @Override
    public boolean initService() {
        layer = taskFlowConfig.getLayers().get(name);
        bucketizer = taskFlow.getLayerBucketizer(layer);
        if (bucketizer == null) {
            log.error("layer bucketizer：{} init fail！", layer.getBucketizer());
            return false;
        }
        return true;
    }

    @Override
    public ServiceRequest makeRequest(String depend, ServiceRequest request, DataContext context) {
        ServiceRequest req = super.makeRequest(depend, request, context);
        if (MapUtils.isNotEmpty(layer.getOptions())) {
            req.getData().putAll(layer.getOptions());
        }
        return req;
    }

    @Override
    public DataResult process(ServiceRequest request, DataContext context) {
        DataResult result = null;
        String experiment = bucketizer.toBucket(context);
        DataService dataService = taskServices.get(experiment);
        if (dataService == null) {
            log.error("layer:{} experiment:{} service init fail!", name, experiment);
            context.setStatus(name, TaskStatusEnum.DEPEND_INIT_FAIL);
            return result;
        }
        ServiceRequest taskRequest = makeRequest(experiment, request, context);
        if (taskRequest == null) {
            log.error("layer:{} experiment:{} request init fail!", name, experiment);
            context.setStatus(name, TaskStatusEnum.DEPEND_INIT_FAIL);
            return result;
        }
        dataService.execute(taskRequest, context);
        result = new DataResult();
        List<Map> data = getTaskResultByColumns(List.of(experiment), false, layer.getColumnNames(), context);
        if (data == null) {
            log.error("layer:{} experiment:{} get result fail!", name, experiment);
            context.setStatus(name, TaskStatusEnum.EXEC_FAIL);
            return result;
        }
        result.setData(data);
        context.setStatus(name, TaskStatusEnum.SUCCESS);
        return result;
    }
}
