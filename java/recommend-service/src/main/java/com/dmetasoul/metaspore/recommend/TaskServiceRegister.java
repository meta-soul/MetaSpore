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
import com.dmetasoul.metaspore.recommend.annotation.FunctionAnnotation;
import com.dmetasoul.metaspore.recommend.annotation.ServiceAnnotation;
import com.dmetasoul.metaspore.recommend.bucketizer.LayerBucketizer;
import com.dmetasoul.metaspore.recommend.common.SpringBeanUtil;
import com.dmetasoul.metaspore.recommend.configure.FeatureConfig;
import com.dmetasoul.metaspore.recommend.configure.RecommendConfig;
import com.dmetasoul.metaspore.recommend.configure.TaskFlowConfig;
import com.dmetasoul.metaspore.recommend.dataservice.*;
import com.dmetasoul.metaspore.recommend.functions.Function;
import com.dmetasoul.metaspore.recommend.datasource.DataSource;
import com.dmetasoul.metaspore.recommend.recommend.Experiment;
import com.dmetasoul.metaspore.recommend.recommend.Layer;
import com.dmetasoul.metaspore.recommend.recommend.Scene;
import com.dmetasoul.metaspore.recommend.recommend.Service;
import com.google.common.collect.Maps;
import lombok.Data;
import lombok.SneakyThrows;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.StringUtils;
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
    @Autowired
    private ExecutorService taskPool;
    /**
     * 用于注册保存各种DataService，每次更新配置，重新生成；
     */
    private Map<String, DataService> dataServices;
    /**
     * 用于注册保存各种Experiment，每次更新配置，类似于dataServices
     */
    private Map<String, Experiment> experimentMap;
    /**
     * 用于注册保存各种Layer，每次更新配置，类似于dataServices
     */
    private Map<String, Layer> layerMap;
    /**
     * 用于注册保存各种Scene，每次更新配置，类似于dataServices
     */
    private Map<String, Scene> sceneMap;
    /**
     * 用于注册保存各种Service，每次更新配置，类似于dataServices
     */
    private Map<String, Service> recommendServices;
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
    @SneakyThrows
    @PostConstruct
    public void initServices() {
        initDataSources();
        initFunctions();
        initDataService();
        initRecommendServices();
        initExperimentMap();
        initLayerMap();
        initSceneMap();
    }

    /**
     * 每次refresh配置后首先destroy所有注册的bean
     */
    @PreDestroy
    public void destroy() {
        dataSources.forEach((name,source)->source.close());
        dataServices.forEach((name,service)->service.close());
        recommendServices.forEach((name,service)->service.close());
        experimentMap.forEach((name,service)->service.close());
        layerMap.forEach((name,service)->service.close());
        sceneMap.forEach((name,service)->service.close());
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
            if (bean == null || !bean.getClass().isAnnotationPresent(ServiceAnnotation.class)) {
                log.error("the datasource kind:{} load fail!", kind);
                throw new RuntimeException(String.format("the datasource kind:%s load fail!", kind));
            }
            dataSources.put(name, bean);
            dataSources.get(name).init(name, taskFlowConfig, sourcePool);
        }
    }

    public DataSource getDataSource(String name) {
        return dataSources.get(name);
    }

    public DataService getDataService(String name) {
        return dataServices.get(name);
    }
    /**
     * 根据最新的配置初始化DataService
     */
    public void initDataService() {
        dataServices = Maps.newHashMap();
        taskFlowConfig.getSourceTables().forEach((name, config) -> {
            SourceTableTask task = (SourceTableTask) SpringBeanUtil.getBean(config.getTaskName());
            if (task == null) {
                task = (SourceTableTask) SpringBeanUtil.getBean("SourceTable");
            }
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
            AlgoTransformTask task = (AlgoTransformTask) SpringBeanUtil.getBean(config.getName());
            if (task == null) {
                if (StringUtils.isNotEmpty(config.getTaskName())) {
                    task = (AlgoTransformTask) SpringBeanUtil.getBean(config.getTaskName());
                }
                if (task == null) {
                    log.error("the AlgoTransformTask:{} load fail!", name);
                    throw new RuntimeException(String.format("the AlgoTransformTask:%s load fail!", name));
                }
            }
            dataServices.put(name, task);
            if (!dataServices.get(name).init(name, taskFlowConfig, this, workFlowPool)) {
                log.error("the AlgoTransformTask:{} init fail!", name);
                throw new RuntimeException(String.format("the AlgoTransformTask:%s init fail!", name));
            }
        });
    }
    /**
     * 根据最新的配置初始化Function
     */
    public void initFunctions() {
        functions = Maps.newHashMap();
        Map<String, Object> beanMap = SpringBeanUtil.getBeanMapByAnnotation(FunctionAnnotation.class);
        for (Map.Entry<String, Object> entry : beanMap.entrySet()) {
            String name = entry.getKey();
            Object func = entry.getValue();
            if (!Function.class.isAssignableFrom(func.getClass())) {
                log.error("the bean :{} load fail, is not instance of Function!", name);
                throw new RuntimeException(String.format("the Function:%s load fail!", name));
            }
            functions.put(name, (Function)func);
        }
    }
    public Service getRecommendService(String name) {
        return recommendServices.get(name);
    }
    /**
     * 根据最新的配置初始化RecommendService
     */
    public void initRecommendServices() {
        recommendServices = Maps.newHashMap();
        taskFlowConfig.getServices().forEach((name, service) -> {
            Service recommendService = (Service) SpringBeanUtil.getBean(service.getName());
            if (recommendService == null) {
                recommendService = (Service) SpringBeanUtil.getBean(service.getTaskName());
                if (recommendService == null) {
                    log.error("the RecommendService:{} load fail!", service.getName());
                    throw new RuntimeException(String.format("the RecommendService:%s load fail!", service.getName()));
                }
            }
            recommendService.init(name, taskFlowConfig, this);
            recommendServices.put(service.getName(), recommendService);
        });
    }

    public Experiment getExperiment(String name) {
        return experimentMap.get(name);
    }

    public void initExperimentMap() {
        experimentMap = Maps.newHashMap();
        taskFlowConfig.getExperiments().forEach((name, experiment) -> {
            Experiment experimentService = (Experiment) SpringBeanUtil.getBean(experiment.getName());
            if (experimentService == null) {
                experimentService = (Experiment) SpringBeanUtil.getBean(experiment.getTaskName());
                if (experimentService == null) {
                    log.error("the RecommendService:{} load fail!", experiment.getName());
                    throw new RuntimeException(String.format("the RecommendService:%s load fail!", experiment.getName()));
                }
            }
            experimentService.init(name, taskFlowConfig, this);
            experimentMap.put(experiment.getName(), experimentService);
        });
    }
    public Layer getLayer(String name) {
        return layerMap.get(name);
    }
    public void initLayerMap() {
        layerMap = Maps.newHashMap();
        taskFlowConfig.getLayers().forEach((name, layer) -> {
            Layer layerService = (Layer) SpringBeanUtil.getBean(layer.getName());
            if (layerService == null) {
                layerService = (Layer) SpringBeanUtil.getBean(layer.getTaskName());
                if (layerService == null) {
                    log.error("the RecommendService:{} load fail!", layer.getName());
                    throw new RuntimeException(String.format("the RecommendService:%s load fail!", layer.getName()));
                }
            }
            layerService.init(name, taskFlowConfig, this);
            layerMap.put(layer.getName(), layerService);
        });
    }
    public Scene getScene(String name) {
        return sceneMap.get(name);
    }
    public void initSceneMap() {
        sceneMap = Maps.newHashMap();
        taskFlowConfig.getScenes().forEach((name, scene) -> {
            Scene sceneService = (Scene) SpringBeanUtil.getBean(scene.getName());
            if (sceneService == null) {
                sceneService = (Scene) SpringBeanUtil.getBean(scene.getTaskName());
                if (sceneService == null) {
                    log.error("the RecommendService:{} load fail!", scene.getName());
                    throw new RuntimeException(String.format("the RecommendService:%s load fail!", scene.getName()));
                }
            }
            sceneService.init(name, taskFlowConfig, this);
            sceneMap.put(scene.getName(), sceneService);
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
