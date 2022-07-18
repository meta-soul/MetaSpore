package com.dmetasoul.metaspore.recommend;

import com.dmetasoul.metaspore.recommend.annotation.BucketizerAnnotation;
import com.dmetasoul.metaspore.recommend.annotation.DataSourceAnnotation;
import com.dmetasoul.metaspore.recommend.annotation.RecommendAnnotation;
import com.dmetasoul.metaspore.recommend.annotation.TransformFunction;
import com.dmetasoul.metaspore.recommend.common.SpringBeanUtil;
import com.dmetasoul.metaspore.recommend.configure.FeatureConfig;
import com.dmetasoul.metaspore.recommend.configure.FunctionConfig;
import com.dmetasoul.metaspore.recommend.configure.RecommendConfig;
import com.dmetasoul.metaspore.recommend.configure.TaskFlowConfig;
import com.dmetasoul.metaspore.recommend.dataservice.*;
import com.dmetasoul.metaspore.recommend.bucketizer.LayerBucketizer;
import com.dmetasoul.metaspore.recommend.functions.Function;
import com.dmetasoul.metaspore.recommend.recommend.RecommendService;
import com.dmetasoul.metaspore.recommend.datasource.DataSource;
import com.google.common.collect.Maps;
import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.cloud.context.config.annotation.RefreshScope;
import org.springframework.stereotype.Component;

import javax.annotation.PostConstruct;
import java.util.Map;
import java.util.concurrent.ExecutorService;

@Slf4j
@Data
@Component
public class TaskFlow {

    @Autowired
    public TaskFlowConfig taskFlowConfig;

    @Autowired
    private SpringBeanUtil springBeanUtil;

    @Autowired
    public ExecutorService featurePool;

    @Autowired
    public ExecutorService taskPool;

    @Autowired
    public ExecutorService workFlowPool;

    private Map<String, DataService> taskServices;

    private Map<String, DataSource> dataSources;
    private Map<String, Function> functions;

    @PostConstruct
    public void init() {
        if (!initDataSource()) {
            log.error("load datasource fail!");
            throw new RuntimeException("load datasource fail!");
        }
        if (!initFunction()) {
            log.error("load function fail!");
            throw new RuntimeException("load function fail!");
        }
        if (!initDataService()) {
            log.error("load dataservice fail!");
            throw new RuntimeException("load dataservice fail!");
        }
    }

    @RefreshScope
    public boolean initDataSource() {
        dataSources = Maps.newHashMap();
        Map<String, FeatureConfig.Source> sources = taskFlowConfig.getSources();
        for (Map.Entry<String, FeatureConfig.Source> entry : sources.entrySet()) {
            String name = entry.getKey();
            String kind = entry.getValue().getKind();
            DataSource bean = (DataSource) SpringBeanUtil.getBean(kind);
            if (bean == null || !bean.getClass().isAnnotationPresent(DataSourceAnnotation.class)) {
                log.error("the datasource kind:{} load fail!", kind);
                return false;
            }
            bean.init(name, taskFlowConfig, featurePool);
            dataSources.put(name, bean);
        }
        return true;
    }

    public DataService getTaskService(String name) {
        return taskServices.get(name);
    }
    public boolean initDataService() {
        taskServices = Maps.newHashMap();
        taskFlowConfig.getSourceTables().forEach((name, config) -> {
            SourceTableTask task = SpringBeanUtil.getBean(SourceTableTask.class);
            if (task == null) {
                log.error("the sourceTableTask:{} load fail!", name);
                return;
            }
            if (!task.init(name, taskFlowConfig, this, workFlowPool)) {
                log.error("the sourceTableTask:{} init fail!", name);
                return;
            }
            taskServices.put(name, task);
        });
        taskFlowConfig.getFeatures().forEach((name, config) -> {
            FeatureTask task = SpringBeanUtil.getBean(FeatureTask.class);
            if (task == null) {
                log.error("the FeatureTask:{} load fail!", name);
                return;
            }
            if (!task.init(name, taskFlowConfig, this, workFlowPool)) {
                log.error("the FeatureTask:{} init fail!", name);
                return;
            }
            taskServices.put(name, task);
        });
        taskFlowConfig.getAlgoTransforms().forEach((name, config) -> {
            AlgoTransformTask task = SpringBeanUtil.getBean(AlgoTransformTask.class);
            if (task == null) {
                log.error("the AlgoTransformTask:{} load fail!", name);
                return;
            }
            if (!task.init(name, taskFlowConfig, this, workFlowPool)) {
                log.error("the AlgoTransformTask:{} init fail!", name);
                return;
            }
            taskServices.put(name, task);
        });
        taskFlowConfig.getServices().forEach((name, config) -> {
            ServiceTask task = SpringBeanUtil.getBean(ServiceTask.class);
            if (task == null) {
                log.error("the ServiceTask:{} load fail!", name);
                return;
            }
            if (!task.init(name, taskFlowConfig, this, workFlowPool)) {
                log.error("the ServiceTask:{} init fail!", name);
                return;
            }
            taskServices.put(name, task);
        });
        taskFlowConfig.getExperiments().forEach((name, config) -> {
            ExperimentTask task = SpringBeanUtil.getBean(ExperimentTask.class);
            if (task == null) {
                log.error("the ExperimentTask:{} load fail!", name);
                return;
            }
            if (!task.init(name, taskFlowConfig, this, workFlowPool)) {
                log.error("the ExperimentTask:{} init fail!", name);
                return;
            }
            taskServices.put(name, task);
        });
        taskFlowConfig.getLayers().forEach((name, config) -> {
            LayerTask task = SpringBeanUtil.getBean(LayerTask.class);
            if (task == null) {
                log.error("the LayerTask:{} load fail!", name);
                return;
            }
            if (!task.init(name, taskFlowConfig, this, workFlowPool)) {
                log.error("the LayerTask:{} init fail!", name);
                return;
            }
            taskServices.put(name, task);
        });
        taskFlowConfig.getScenes().forEach((name, config) -> {
            SceneTask task = SpringBeanUtil.getBean(SceneTask.class);
            if (task == null) {
                log.error("the SceneTask:{} load fail!", name);
                return;
            }
            if (!task.init(name, taskFlowConfig, this, workFlowPool)) {
                log.error("the SceneTask:{} init fail!", name);
                return;
            }
            taskServices.put(name, task);
        });
        return true;
    }

    public boolean initFunction() {
        functions = Maps.newHashMap();
        Map<String, Object> beanMap = SpringBeanUtil.getBeanMapByAnnotation(TransformFunction.class);
        for (Map.Entry<String, Object> entry : beanMap.entrySet()) {
            String name = entry.getKey();
            Object func = entry.getValue();
            if (!Function.class.isAssignableFrom(func.getClass())) {
                log.error("the bean :{} load fail, is not instance of Function!", name);
                return false;
            }
            functions.put(name, (Function)func);
            Map<String, Object> params = null;
            FunctionConfig.Function function = taskFlowConfig.getFunctionConfig().getFunctionMap().get(name);
            if (function != null) {
                params = function.getOptions();
            }
            functions.get(name).init(params);
        }
        return true;
    }

    public RecommendService getRecommendService(RecommendConfig.Service service) {
        RecommendService recommendService = (RecommendService) SpringBeanUtil.getBean(service.getServiceName());
        if (recommendService == null || !recommendService.getClass().isAnnotationPresent(RecommendAnnotation.class)) {
            log.error("the RecommendService:{} load fail!", service.getServiceName());
            return null;
        }
        return recommendService;
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
}
