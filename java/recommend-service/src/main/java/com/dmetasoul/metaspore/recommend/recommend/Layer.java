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
package com.dmetasoul.metaspore.recommend.recommend;

import com.dmetasoul.metaspore.recommend.TaskServiceRegister;
import com.dmetasoul.metaspore.recommend.annotation.BucketizerAnnotation;
import com.dmetasoul.metaspore.recommend.annotation.ServiceAnnotation;
import com.dmetasoul.metaspore.recommend.common.SpringBeanUtil;
import com.dmetasoul.metaspore.recommend.configure.RecommendConfig;
import com.dmetasoul.metaspore.recommend.configure.TaskFlowConfig;
import com.dmetasoul.metaspore.recommend.data.DataContext;
import com.dmetasoul.metaspore.recommend.data.DataResult;
import com.dmetasoul.metaspore.recommend.bucketizer.LayerBucketizer;
import com.dmetasoul.metaspore.recommend.recommend.interfaces.BaseService;
import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.StringUtils;
import org.springframework.util.Assert;

import java.util.List;
import java.util.concurrent.CompletableFuture;

@Data
@Slf4j
@ServiceAnnotation("Layer")
public class Layer implements BaseService {
    protected String name;
    protected RecommendConfig.Layer layer;
    protected LayerBucketizer bucketizer;
    protected TaskFlowConfig taskFlowConfig;
    protected TaskServiceRegister serviceRegister;
    public void init(String name, TaskFlowConfig taskFlowConfig, TaskServiceRegister serviceRegister) {
        if (StringUtils.isEmpty(name)) {
            log.error("name is null, init fail!");
        }
        this.name = name;
        this.taskFlowConfig = taskFlowConfig;
        this.serviceRegister = serviceRegister;
        layer = taskFlowConfig.getLayers().get(name);
        bucketizer = getLayerBucketizer(layer);
        if (bucketizer == null) {
            log.error("layer bucketizer：{} init fail！", layer.getBucketizer());
            throw new RuntimeException("layer bucketizer init fail at:" + layer.getBucketizer());
        }
    }
    public LayerBucketizer getLayerBucketizer(RecommendConfig.Layer layer) {
        LayerBucketizer layerBucketizer = (LayerBucketizer) SpringBeanUtil.getBeanByName(layer.getBucketizer());
        if (layerBucketizer == null || !layerBucketizer.getClass().isAnnotationPresent(BucketizerAnnotation.class)) {
            log.error("the layer.getBucketizer:{} load fail!", layer.getBucketizer());
            return null;
        }
        layerBucketizer.init(layer);
        return layerBucketizer;
    }

    @Override
    public CompletableFuture<List<DataResult>> execute(DataResult data, DataContext context) {
        return execute(List.of(data), context);
    }

    @Override
    public CompletableFuture<List<DataResult>> execute(List<DataResult> data, DataContext context) {
        String experiment = bucketizer.toBucket(context);
        Experiment experimentFlow = serviceRegister.getExperiment(experiment);
        Assert.notNull(experimentFlow, "experiment service is not exist! at " + experiment);
        return experimentFlow.process(data, context);
    }

    @Override
    public CompletableFuture<List<DataResult>> execute(DataContext context) {
        return execute(List.of(), context);
    }

    @Override
    public void close() {
    }
}
