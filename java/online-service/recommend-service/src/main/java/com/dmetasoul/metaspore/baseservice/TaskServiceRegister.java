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

package com.dmetasoul.metaspore.baseservice;

import com.dmetasoul.metaspore.FeatureServiceManager;
import com.dmetasoul.metaspore.UDFLoader;
import com.dmetasoul.metaspore.common.SpringBeanUtil;
import com.dmetasoul.metaspore.configure.*;
import com.dmetasoul.metaspore.dataservice.AlgoTransformTask;
import com.dmetasoul.metaspore.dataservice.DataService;
import com.dmetasoul.metaspore.dataservice.FeatureTask;
import com.dmetasoul.metaspore.dataservice.SourceTableTask;
import com.dmetasoul.metaspore.functions.Function;
import com.dmetasoul.metaspore.datasource.DataSource;
import com.dmetasoul.metaspore.recommend.Experiment;
import com.dmetasoul.metaspore.recommend.Layer;
import com.dmetasoul.metaspore.recommend.Scene;
import com.dmetasoul.metaspore.recommend.Service;
import com.google.common.collect.Maps;
import lombok.Data;
import lombok.NonNull;
import lombok.SneakyThrows;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.collections4.CollectionUtils;
import org.apache.commons.collections4.MapUtils;
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

    private FeatureServiceManager featureServiceManager;
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

    private long timestamp;

    /**
     * 每次refresh配置，重新注册生成所有的服务任务bean实例
     */
    @SneakyThrows
    @PostConstruct
    public void initServices() {
        initFeatureService();
        initDataSource();
        initDataService();
        initRecommendServices();
        initExperimentMap();
        initLayerMap();
        initSceneMap();
    }

    public void initFeatureService() {
        this.featureServiceManager = new FeatureServiceManager();
        featureServiceManager.scanClass();
        FunctionConfig functionConfig = taskFlowConfig.getFunctionConfig();
        log.info("functionConfig: {}", functionConfig);
        UDFLoader udfLoader = featureServiceManager.getUdfLoader();
        if (functionConfig != null && functionConfig.getPath() != null
                && CollectionUtils.isNotEmpty(functionConfig.getJars())) {
            for (int i = 0; i < functionConfig.getJars().size(); ++i) {
                FunctionConfig.JarInfo jarInfo = functionConfig.getJars().get(i);
                udfLoader.addJarURL(functionConfig.getPath(), jarInfo.getName());
            }
            for (FunctionConfig.JarInfo jarInfo : functionConfig.getJars()) {
                loadJarInfo(udfLoader, jarInfo);
            }
        }
    }

    public void loadJarInfo(UDFLoader udfLoader, @NonNull FunctionConfig.JarInfo jarInfo) {
        udfLoader.registerUDF(jarInfo.getFieldFunction());
        udfLoader.registerUDF(jarInfo.getTransformFunction());
        udfLoader.registerUDF(jarInfo.getTransformMergeOperator());
        udfLoader.registerUDF(jarInfo.getTransformUpdateOperator());
        udfLoader.registerUDF(jarInfo.getLayerBucketizer());
    }

    /**
     * 每次refresh配置后首先destroy所有注册的bean
     */
    @SneakyThrows
    @PreDestroy
    public void destroy() {
        recommendServices.forEach((name, service) -> service.close());
        experimentMap.forEach((name, service) -> service.close());
        layerMap.forEach((name, service) -> service.close());
        sceneMap.forEach((name, service) -> service.close());
        featureServiceManager.close();
        log.info("refresh destroy bean!");
    }

    @SuppressWarnings("unchecked")
    public <T> T getBean(String name, Class<?> cls, boolean hold) {
        T bean = (T) SpringBeanUtil.getBean(name);
        if (bean == null) {
            bean = featureServiceManager.getBean(name, cls, hold, true);
            if (bean != null) {
                return bean;
            } else {
                log.warn("load bean: {} fail!", name);
            }
        }
        log.info("load bean:{} from spring", name);
        return bean;
    }

    @SuppressWarnings("unchecked")
    private <T> T getBean(String name) {
        return (T) SpringBeanUtil.getBean(name);
    }

    /**
     * 根据最新的配置初始化DataSources
     */

    public Map<String, DataSource> getDataSources() {
        return featureServiceManager.getDataSources();
    }

    public DataService getDataService(String name) {
        return featureServiceManager.getDataService(name);
    }

    public <T> T getUDFBean(String name, Class<?> cls) {
        return featureServiceManager.getUdfLoader().getBean(name, cls, true);
    }

    public void initDataSource() {
        if (taskFlowConfig == null) return;
        if (MapUtils.isNotEmpty(taskFlowConfig.getSources())) {
            taskFlowConfig.getSources().forEach((name, config) -> {
                String kind = config.getKind();
                DataSource dataSource = getBean(kind, DataSource.class, false);
                if (dataSource == null) {
                    log.error("no found the DataSource:{}， {}!", kind, name);
                } else {
                    if (!dataSource.init(name, config, sourcePool)) {
                        log.error("the DataSource:{} init fail!", name);
                        throw new RuntimeException("the DataSource init fail at " + name);
                    }
                    featureServiceManager.addDataSource(name, dataSource);
                }
            });
        }
    }

    public void addDataService(String taskName, String name, TableInfo config, Class<?> cls) {
        DataService dataService = getBean(taskName, cls, false);
        if (dataService == null) {
            log.error("no found the DataService:{} at {}!", taskName, name);
        } else {
            if (!dataService.init(name, config, featureServiceManager, workFlowPool)) {
                log.error("the DataService:{} init fail!", name);
                throw new RuntimeException("the DataService init fail at " + name);
            }
            featureServiceManager.addDataService(name, dataService);
        }
    }

    /**
     * 根据最新的配置初始化DataService
     */
    public void initDataService() {
        if (taskFlowConfig == null) return;
        if (MapUtils.isNotEmpty(taskFlowConfig.getSourceTables())) {
            taskFlowConfig.getSourceTables().forEach((name, config) -> {
                String taskName = config.getTaskName();
                addDataService(taskName, name, config, SourceTableTask.class);
            });
        }
        if (MapUtils.isNotEmpty(taskFlowConfig.getFeatures())) {
            taskFlowConfig.getFeatures().forEach((name, config) -> {
                addDataService("feature", name, config, FeatureTask.class);
            });
        }
        if (MapUtils.isNotEmpty(taskFlowConfig.getAlgoTransforms())) {
            taskFlowConfig.getAlgoTransforms().forEach((name, config) -> {
                String taskName = config.getTaskName();
                addDataService(taskName, name, config, AlgoTransformTask.class);
            });
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
        if (taskFlowConfig == null) return;
        if (MapUtils.isNotEmpty(taskFlowConfig.getServices())) {
            taskFlowConfig.getServices().forEach((name, service) -> {
                Service recommendService = getBean(service.getName());
                if (recommendService == null) {
                    recommendService = getBean(service.getTaskName());
                    if (recommendService == null) {
                        log.error("the RecommendService:{} load fail!", service.getName());
                        throw new RuntimeException(String.format("the RecommendService:%s load fail!", service.getName()));
                    }
                }
                recommendService.init(name, taskFlowConfig, this);
                recommendServices.put(service.getName(), recommendService);
            });
        }
    }

    public Experiment getExperiment(String name) {
        return experimentMap.get(name);
    }

    public void initExperimentMap() {
        experimentMap = Maps.newHashMap();
        if (taskFlowConfig == null) return;
        if (MapUtils.isNotEmpty(taskFlowConfig.getExperiments())) {
            taskFlowConfig.getExperiments().forEach((name, experiment) -> {
                Experiment experimentService = getBean(experiment.getName());
                if (experimentService == null) {
                    experimentService = getBean(experiment.getTaskName());
                    if (experimentService == null) {
                        log.error("the RecommendService:{} load fail!", experiment.getName());
                        throw new RuntimeException(String.format("the RecommendService:%s load fail!", experiment.getName()));
                    }
                }
                experimentService.init(name, taskFlowConfig, this);
                experimentMap.put(experiment.getName(), experimentService);
            });
        }
    }

    public Layer getLayer(String name) {
        return layerMap.get(name);
    }

    public void initLayerMap() {
        layerMap = Maps.newHashMap();
        if (taskFlowConfig == null) return;
        if (MapUtils.isNotEmpty(taskFlowConfig.getLayers())) {
            taskFlowConfig.getLayers().forEach((name, layer) -> {
                Layer layerService = getBean(layer.getName());
                if (layerService == null) {
                    layerService = getBean(layer.getTaskName());
                    if (layerService == null) {
                        log.error("the RecommendService:{} load fail!", layer.getName());
                        throw new RuntimeException(String.format("the RecommendService:%s load fail!", layer.getName()));
                    }
                }
                layerService.init(name, taskFlowConfig, this);
                layerMap.put(layer.getName(), layerService);
            });
        }
    }

    public Scene getScene(String name) {
        return sceneMap.get(name);
    }

    public void initSceneMap() {
        sceneMap = Maps.newHashMap();
        if (taskFlowConfig == null) return;
        if (MapUtils.isNotEmpty(taskFlowConfig.getScenes())) {
            taskFlowConfig.getScenes().forEach((name, scene) -> {
                Scene sceneService = getBean(scene.getName());
                if (sceneService == null) {
                    sceneService = getBean(scene.getTaskName());
                    if (sceneService == null) {
                        log.error("the RecommendService:{} load fail!", scene.getName());
                        throw new RuntimeException(String.format("the RecommendService:%s load fail!", scene.getName()));
                    }
                }
                sceneService.init(name, taskFlowConfig, this);
                sceneMap.put(scene.getName(), sceneService);
            });
        }
    }
}
