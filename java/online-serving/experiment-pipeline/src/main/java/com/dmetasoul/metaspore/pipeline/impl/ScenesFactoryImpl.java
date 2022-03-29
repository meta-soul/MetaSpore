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

package com.dmetasoul.metaspore.pipeline.impl;


import com.dmetasoul.metaspore.pipeline.BaseExperiment;
import com.dmetasoul.metaspore.pipeline.BaseLayer;
import com.dmetasoul.metaspore.pipeline.Scene;
import com.dmetasoul.metaspore.pipeline.ScenesFactory;
import com.dmetasoul.metaspore.pipeline.annotation.ExperimentAnnotation;
import com.dmetasoul.metaspore.pipeline.annotation.LayerAnnotation;
import com.dmetasoul.metaspore.pipeline.pojo.*;
import lombok.Data;
import lombok.SneakyThrows;
import org.springframework.cloud.context.config.annotation.RefreshScope;
import org.springframework.context.ApplicationContext;

import java.util.*;

@Data
@RefreshScope
public class ScenesFactoryImpl implements ScenesFactory {
    private SceneConfig sceneConfig;
    private Map<String, Scene> scenes;

    public ScenesFactoryImpl(SceneConfig sceneConfig, ApplicationContext ctx) {
        this.sceneConfig = sceneConfig;
        this.scenes = initScenes(sceneConfig, ctx);
    }

    @Override
    public Scene getScene(String sceneName) {
        return scenes.get(sceneName);
    }


    @Data
    class LayerBean {
        private String layerName;
        private LayerArgs layerArgs;
        private BaseLayer layerBeanObject;
        private Class<?> inputClass;
        private Map<String, ExperimentBean> experimentBeans = new HashMap<>();

        public ExperimentBean getExperimentBean(String experimentName) {
            return experimentBeans.get(experimentName);
        }

        private void putExperiment(String expeirmentName, float ratio, BaseExperiment experimentObject, Map<String, Object> experimentArgs) {
            experimentBeans.put(expeirmentName, new ExperimentBean(expeirmentName, ratio, experimentObject, experimentArgs));
        }
    }

    @Data
    class ExperimentBean {

        private String experimentName;

        private float ratio;

        private BaseExperiment experimentObject;

        private Map<String, Object> extraExperimentArgs;

        public ExperimentBean(String experimentName, float ratio, BaseExperiment experimentObject, Map<String, Object> extraExperimentArgs) {
            this.experimentName = experimentName;
            this.ratio = ratio;
            this.experimentObject = experimentObject;
            this.extraExperimentArgs = extraExperimentArgs;
        }
    }

    private Map<String, Scene> initScenes(SceneConfig sceneConfig, ApplicationContext ctx) {
        HashMap<String, Scene> scenes = new HashMap<>();
        Map<String, Object> layerBeanMap = ctx.getBeansWithAnnotation(LayerAnnotation.class);
        Map<String, Object> experimentBeanMap = ctx.getBeansWithAnnotation(ExperimentAnnotation.class);
        for (com.dmetasoul.metaspore.pipeline.pojo.Scene scene : sceneConfig.getScenes()) {
            String sceneName = scene.getName();
            // layers
            SceneImpl sceneImplBean = new SceneImpl();
            Map<String, Object> extraSceneArgs = scene.getExtraSceneArgs();
            sceneImplBean.setExtraSceneArgs(extraSceneArgs);
            for (Layer layer : scene.getLayers()) {
                // layerInstance
                String layerName = layer.getName();
                LayerBean layerInstance = new LayerBean();
                BaseLayer layerBeanObject = getLayerBeanObject(layerName, layerBeanMap, ctx);
                List<NormalLayerArgs> normalLayerArgsList = layer.getNormalLayerArgs();
                // extraLayerArgs
                Map<String, Object> extraLayerArgs = new HashMap<>();
                extraLayerArgs.putAll(extraSceneArgs);
                extraLayerArgs.putAll(layer.getExtraLayerArgs());

                LayerArgs layerArgs = new LayerArgs(normalLayerArgsList, extraLayerArgs);
                layerInstance.setLayerName(layerName);
                layerInstance.setLayerBeanObject(layerBeanObject);
                layerInstance.setLayerArgs(layerArgs);
                // layer.intitialize()
                layerBeanObject.intitialize(layerArgs);
                for (NormalLayerArgs normalLayerArgs : normalLayerArgsList) {
                    String experimentName = normalLayerArgs.getExperimentName();
                    float ratio = normalLayerArgs.getRatio();
                    // get experimentConfig
                    Experiment experimentConfig = getExperimentConfig(layerName, experimentName);
                    // experimentBeanObject
                    BaseExperiment experimentBeanObject = getExperimentBeanObject(experimentName, experimentConfig, experimentBeanMap, ctx);
                    // extraExperimentArgs
                    Map<String, Object> extraExperimentArgs = new HashMap<>();
                    extraExperimentArgs.putAll(extraLayerArgs);
                    extraExperimentArgs.putAll(experimentConfig.getExtraExperimentArgs());
                    experimentBeanObject.initialize(extraExperimentArgs);
                    // experiment initialize()
                    layerInstance.putExperiment(experimentName, ratio, experimentBeanObject, extraExperimentArgs);
                }
                sceneImplBean.add(layerInstance);
            }
            scenes.put(sceneName, sceneImplBean);
        }
        return scenes;
    }


    private Experiment getExperimentConfig(String layerName, String experimentName) {
        Optional<Experiment> experimentConfig = sceneConfig.getExperiments().stream().filter(x -> x.getLayerName().equals(layerName) && x.getExperimentName().equals(experimentName)).findFirst();
        if (experimentConfig.isPresent()) {
            return experimentConfig.get();
        } else {
            throw new RuntimeException(String.format("experiment: %s of layer %s was not founded in experiment config, please check it again", experimentName, layerName));
        }

    }

    private BaseLayer getLayerBeanObject(String layerName, Map<String, Object> layerBeanMap, ApplicationContext ctx) {
        System.out.println("layerName: " + layerName);
        System.out.println("layerBeanMap: " + layerBeanMap);

        Optional<Map.Entry<String, Object>> layerBeanObject = layerBeanMap.entrySet().stream().filter(map -> ctx.findAnnotationOnBean(map.getKey(), LayerAnnotation.class).name().equals(layerName)).findFirst();

        if (layerBeanObject.isPresent()) {
            String layerBeanName = layerBeanObject.get().getKey();
            return (BaseLayer) layerBeanMap.get(layerBeanName);
        } else {
            throw new RuntimeException(String.format("%s layer should be implemented by BaseLayer interface", layerName));

        }
    }

//    private LayerBean getlayerInstance() {
//        LayerBean layerInstance = new LayerBean();
//        layerInstance.setLayerName(layerName);
//        layerInstance.setLayerBeanObject(layerBeanObject);
//        layerInstance.setLayerArgs(layerArgs);
//    }

    private BaseExperiment getExperimentBeanObject(String experimentName, Experiment experimentConfig, Map<String, Object> experimentBeanMap, ApplicationContext ctx) {
        String classNameArg = experimentConfig.getClassName();

        if (classNameArg == null) {
            return getExperimentBeanObjectFromCtx(experimentName, experimentBeanMap, ctx);
        } else if (checkClassNameArgAvailable(classNameArg)) {
            return getExperimentBeanObjectFromReflect(classNameArg);
        } else {
            throw new RuntimeException(String.format("%s experiment init failed, please check experiment config or implemention of BaseExperiment interface", experimentName));
        }
    }

    private BaseExperiment getExperimentBeanObjectFromCtx(String experimentName, Map<String, Object> experimentBeanMap, ApplicationContext ctx) {

        Optional<Map.Entry<String, Object>> experimentBeanObject = experimentBeanMap.entrySet().stream().filter(map -> ctx.findAnnotationOnBean(map.getKey(), ExperimentAnnotation.class).name().equals(experimentName)).findFirst();
        if (experimentBeanObject.isPresent()) {
            String experimentBeanName = experimentBeanObject.get().getKey();
            return (BaseExperiment) experimentBeanMap.get(experimentBeanName);
        } else {
            throw new RuntimeException(String.format("%s experiment should be implemented by BaseExperiment interface", experimentName));
        }
    }

    @SneakyThrows
    private BaseExperiment getExperimentBeanObjectFromReflect(String experimentClassName) {
        BaseExperiment experimentInstance = null;
        try {
            Class<?> experimentClazz = Class.forName(experimentClassName);
            experimentInstance = (BaseExperiment) experimentClazz.newInstance();
            System.out.println(experimentInstance);
        } catch (ClassNotFoundException | InstantiationException | IllegalAccessException e) {
            e.printStackTrace();
        }
        return experimentInstance;
    }

    // TODO: 2022/3/16 增加上下层 input/output 类型校验
    private void checkInOutClassType(Class<?> a, Class<?> b) {
        if (a.isInstance(b)) {
        }
    }

    // TODO: 2022/3/29 校验 classNameArg
    private boolean checkClassNameArgAvailable(String classNameArg) {
        return true;
    }


}