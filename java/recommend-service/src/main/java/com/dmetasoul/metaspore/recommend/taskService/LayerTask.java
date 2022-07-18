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
package com.dmetasoul.metaspore.recommend.taskService;

import com.dmetasoul.metaspore.recommend.annotation.BucketizerAnnotation;
import com.dmetasoul.metaspore.recommend.annotation.DataServiceAnnotation;
import com.dmetasoul.metaspore.recommend.common.SpringBeanUtil;
import com.dmetasoul.metaspore.recommend.configure.RecommendConfig;
import com.dmetasoul.metaspore.recommend.configure.TaskFlowConfig;
import com.dmetasoul.metaspore.recommend.data.DataContext;
import com.dmetasoul.metaspore.recommend.data.DataResult;
import com.dmetasoul.metaspore.recommend.data.ServiceRequest;
import com.dmetasoul.metaspore.recommend.dataservice.DataService;
import com.dmetasoul.metaspore.recommend.enums.TaskStatusEnum;
import com.dmetasoul.metaspore.recommend.bucketizer.LayerBucketizer;
import com.google.common.collect.Maps;
import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.collections4.MapUtils;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import java.util.List;
import java.util.Map;

@SuppressWarnings("rawtypes")
@Data
@Slf4j
@DataServiceAnnotation("Layer")
public class LayerTask extends TaskService {

    private RecommendConfig.Layer layer;
    private LayerBucketizer bucketizer;

    @Override
    public boolean initService() {
        layer = taskFlowConfig.getLayers().get(name);
        bucketizer = getLayerBucketizer(layer);
        if (bucketizer == null) {
            log.error("layer bucketizer：{} init fail！", layer.getBucketizer());
            return false;
        }
        return true;
    }

    public LayerBucketizer getLayerBucketizer(RecommendConfig.Layer layer) {
        LayerBucketizer layerBucketizer = (LayerBucketizer) SpringBeanUtil.getBean(layer.getBucketizer());
        if (layerBucketizer == null || !layerBucketizer.getClass().isAnnotationPresent(BucketizerAnnotation.class)) {
            log.error("the layer.getBucketizer:{} load fail!", layer.getBucketizer());
            return null;
        }
        layerBucketizer.init(layer);
        return layerBucketizer;
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
        TaskService taskService = taskServices.get(experiment);
        if (taskService == null) {
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
        taskService.execute(taskRequest, context);
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
