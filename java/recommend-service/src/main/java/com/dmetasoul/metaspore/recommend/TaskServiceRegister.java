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

package com.dmetasoul.metaspore.recommend;

import com.dmetasoul.metaspore.recommend.annotation.BucketizerAnnotation;
import com.dmetasoul.metaspore.recommend.annotation.DataSourceAnnotation;
import com.dmetasoul.metaspore.recommend.annotation.RecommendAnnotation;
import com.dmetasoul.metaspore.recommend.annotation.TransformFunction;
import com.dmetasoul.metaspore.recommend.bucketizer.LayerBucketizer;
import com.dmetasoul.metaspore.recommend.common.SpringBeanUtil;
import com.dmetasoul.metaspore.recommend.configure.FeatureConfig;
import com.dmetasoul.metaspore.recommend.configure.FunctionConfig;
import com.dmetasoul.metaspore.recommend.configure.RecommendConfig;
import com.dmetasoul.metaspore.recommend.configure.TaskFlowConfig;
import com.dmetasoul.metaspore.recommend.dataservice.*;
import com.dmetasoul.metaspore.recommend.functions.Function;
import com.dmetasoul.metaspore.recommend.recommend.RecommendService;
import com.dmetasoul.metaspore.recommend.datasource.DataSource;
import com.dmetasoul.metaspore.recommend.taskService.*;
import com.google.common.collect.Maps;
import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.cloud.context.config.annotation.RefreshScope;
import org.springframework.stereotype.Component;

import javax.annotation.PostConstruct;
import javax.annotation.PreDestroy;
import java.util.Map;
import java.util.concurrent.ExecutorService;

/**
 * 用于注册各种任务的实例；随配置动态更新
 * Created by @author qinyy907 in 14:24 22/07/15.
 */
@Slf4j
@Data
@RefreshScope
@Component
public class TaskServiceRegister {
    /**
     * spring 配置类实例
     */
    @Autowired
    private TaskFlowConfig taskFlowConfig;
    /**
     * spring bean获取工具类实例
     */
    @Autowired
    private SpringBeanUtil springBeanUtil;
    @Autowired
    private ExecutorService sourcePool;
    @Autowired
    private ExecutorService workFlowPool;
    /**
     * 用于注册保存各种DataService，每次更新配置，重新生成；
     */
    private Map<String, DataService> dataServices;
    /**
     * 用于注册保存各种TaskService，每次更新配置，类似于dataServices
     */
    private Map<String, TaskService> taskServices;
    /**
     * 用于注册保存各种RecommendService，每次更新配置，类似于dataServices
     */
    private Map<String, RecommendService> recommendServices;
    /**
     * 用于注册保存各种DataSource，每次更新配置，类似于dataServices
     */
    private Map<String, DataSource> dataSources;
    /**
     * 用于注册保存各种Function，每次更新配置，类似于dataServices
     */
    private Map<String, Function> functions;

    /**
     * 每次refresh配置，重新注册生成所有的服务任务bean实例
     */
    @PostConstruct
    public void initServices() {
        initDataSources();
        initDataService();
        initTaskServices();
        initFunctions();
        initRecommendServices();
    }

    /**
     * 每次refresh配置后首先destroy所有注册的bean
     */
    @PreDestroy
    public void destroy() {
        dataSources.forEach((name,source)->source.close());
        dataServices.forEach((name,service)->service.close());
        taskServices.forEach((name,service)->service.close());
        recommendServices.forEach((name,service)->service.close());
    }

    /**
     * 根据最新的配置初始化DataSources
     */
    public void initDataSources() {
        dataSources = Maps.newHashMap();
        Map<String, FeatureConfig.Source> sources = taskFlowConfig.getSources();
        for (Map.Entry<String, FeatureConfig.Source> entry : sources.entrySet()) {
            String name = entry.getKey();
            String kind = entry.getValue().getKind();
            DataSource bean = (DataSource) SpringBeanUtil.getBean(kind);
            if (bean == null || !bean.getClass().isAnnotationPresent(DataSourceAnnotation.class)) {
                log.error("the datasource kind:{} load fail!", kind);
                throw new RuntimeException(String.format("the datasource kind:%s load fail!", kind));
            }
            dataSources.put(name, bean);
            dataSources.get(name).init(name, taskFlowConfig, sourcePool);
        }
    }

    public DataService getTaskService(String name) {
        return dataServices.get(name);
    }
    /**
     * 根据最新的配置初始化DataService
     */
    public void initDataService() {
        dataServices = Maps.newHashMap();
        taskFlowConfig.getSourceTables().forEach((name, config) -> {
            SourceTableTask task = SpringBeanUtil.getBean(SourceTableTask.class);
            if (task == null) {
                log.error("the sourceTableTask:{} load fail!", name);
                throw new RuntimeException(String.format("the sourceTableTask:%s load fail!", name));
            }
            dataServices.put(name, task);
            if (!dataServices.get(name).init(name, taskFlowConfig, this, workFlowPool)) {
                log.error("the sourceTableTask:{} init fail!", name);
                throw new RuntimeException(String.format("the sourceTableTask:%s init fail!", name));
            }
        });
        taskFlowConfig.getFeatures().forEach((name, config) -> {
            FeatureTask task = SpringBeanUtil.getBean(FeatureTask.class);
            if (task == null) {
                log.error("the FeatureTask:{} load fail!", name);
                throw new RuntimeException(String.format("the FeatureTask:%s load fail!", name));
            }
            dataServices.put(name, task);
            if (!dataServices.get(name).init(name, taskFlowConfig, this, workFlowPool)) {
                log.error("the FeatureTask:{} init fail!", name);
                throw new RuntimeException(String.format("the FeatureTask:%s init fail!", name));
            }
        });
        taskFlowConfig.getAlgoTransforms().forEach((name, config) -> {
            AlgoTransformTask task = SpringBeanUtil.getBean(AlgoTransformTask.class);
            if (task == null) {
                log.error("the AlgoTransformTask:{} load fail!", name);
                throw new RuntimeException(String.format("the AlgoTransformTask:%s load fail!", name));
            }
            dataServices.put(name, task);
            if (!dataServices.get(name).init(name, taskFlowConfig, this, workFlowPool)) {
                log.error("the AlgoTransformTask:{} init fail!", name);
                throw new RuntimeException(String.format("the AlgoTransformTask:%s init fail!", name));
            }
        });
    }
    /**
     * 根据最新的配置初始化TaskService
     */
    public void initTaskServices() {
        taskServices = Maps.newHashMap();
        taskFlowConfig.getServices().forEach((name, config) -> {
            ServiceTask task = SpringBeanUtil.getBean(ServiceTask.class);
            if (task == null) {
                log.error("the ServiceTask:{} load fail!", name);
                throw new RuntimeException(String.format("the ServiceTask:%s load fail!", name));
            }
            taskServices.put(name, task);
            if (!taskServices.get(name).init(name, taskFlowConfig, this, workFlowPool)) {
                log.error("the ServiceTask:{} init fail!", name);
                throw new RuntimeException(String.format("the ServiceTask:%s init fail!", name));
            }
        });
        taskFlowConfig.getExperiments().forEach((name, config) -> {
            ExperimentTask task = SpringBeanUtil.getBean(ExperimentTask.class);
            if (task == null) {
                log.error("the ExperimentTask:{} load fail!", name);
                throw new RuntimeException(String.format("the ExperimentTask:%s load fail!", name));
            }
            taskServices.put(name, task);
            if (!taskServices.get(name).init(name, taskFlowConfig, this, workFlowPool)) {
                log.error("the ExperimentTask:{} init fail!", name);
                throw new RuntimeException(String.format("the ExperimentTask:%s init fail!", name));
            }
        });
        taskFlowConfig.getLayers().forEach((name, config) -> {
            LayerTask task = SpringBeanUtil.getBean(LayerTask.class);
            if (task == null) {
                log.error("the LayerTask:{} load fail!", name);
                throw new RuntimeException(String.format("the LayerTask:%s load fail!", name));
            }
            taskServices.put(name, task);
            if (!taskServices.get(name).init(name, taskFlowConfig, this, workFlowPool)) {
                log.error("the LayerTask:{} init fail!", name);
                throw new RuntimeException(String.format("the LayerTask:%s init fail!", name));
            }
        });
        taskFlowConfig.getScenes().forEach((name, config) -> {
            SceneTask task = SpringBeanUtil.getBean(SceneTask.class);
            if (task == null) {
                log.error("the SceneTask:{} load fail!", name);
                throw new RuntimeException(String.format("the SceneTask:%s load fail!", name));
            }
            taskServices.put(name, task);
            if (!taskServices.get(name).init(name, taskFlowConfig, this, workFlowPool)) {
                log.error("the SceneTask:{} init fail!", name);
                throw new RuntimeException(String.format("the SceneTask:%s init fail!", name));
            }
        });
    }
    /**
     * 根据最新的配置初始化Function
     */
    public void initFunctions() {
        functions = Maps.newHashMap();
        Map<String, Object> beanMap = SpringBeanUtil.getBeanMapByAnnotation(TransformFunction.class);
        for (Map.Entry<String, Object> entry : beanMap.entrySet()) {
            String name = entry.getKey();
            Object func = entry.getValue();
            if (!Function.class.isAssignableFrom(func.getClass())) {
                log.error("the bean :{} load fail, is not instance of Function!", name);
                throw new RuntimeException(String.format("the Function:%s load fail!", name));
            }
            functions.put(name, (Function)func);
            Map<String, Object> params = null;
            FunctionConfig.Function function = taskFlowConfig.getFunctionConfig().getFunctionMap().get(name);
            if (function != null) {
                params = function.getOptions();
            }
            functions.get(name).init(params);
        }
    }

    /**
     * 根据最新的配置初始化RecommendService
     */
    public void initRecommendServices() {
        recommendServices = Maps.newHashMap();
        taskFlowConfig.getServices().forEach((name, service) -> {
            RecommendService recommendService = (RecommendService) SpringBeanUtil.getBean(service.getServiceName());
            if (recommendService == null || !recommendService.getClass().isAnnotationPresent(RecommendAnnotation.class)) {
                log.error("the RecommendService:{} load fail!", service.getServiceName());
                throw new RuntimeException(String.format("the RecommendService:%s load fail!", service.getServiceName()));
            }
            recommendServices.put(service.getServiceName(), recommendService);
        });
    }

    public LayerBucketizer getLayerBucketizer(RecommendConfig.Layer layer) {
        LayerBucketizer layerBucketizer = (LayerBucketizer) SpringBeanUtil.getBean(layer.getBucketizer());
        if (layerBucketizer == null || !layerBucketizer.getClass().isAnnotationPresent(BucketizerAnnotation.class)) {
            log.error("the layer.getBucketizer:{} load fail!", layer.getBucketizer());
            throw new RuntimeException(String.format("the layer.getBucketizer:%s load fail!", layer.getBucketizer()));
        }
        layerBucketizer.init(layer);
        return layerBucketizer;
    }
}
