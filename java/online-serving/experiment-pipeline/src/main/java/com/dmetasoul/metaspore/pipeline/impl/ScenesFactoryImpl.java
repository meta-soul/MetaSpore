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
//import com.dmetasoul.metaspore.pipeline.utils.ApplicationContextGetBeanHelper;
import lombok.Data;
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
        private BaseLayer layerClass;
        private Class<?> inputClass;
        private Map<String, ExperimentBean> experimentBeans = new HashMap<>();

        public ExperimentBean getExperimentBean(String experimentName) {
            return experimentBeans.get(experimentName);
        }

        private void putExperiment(String expeirmentName, float ratio, BaseExperiment experimentCls, Map<String, Object> experimentArgs) {
            experimentBeans.put(expeirmentName, new ExperimentBean(expeirmentName, ratio, experimentCls, experimentArgs));
        }
    }

    @Data
    class ExperimentBean {

        private String experimentName;

        private float ratio;

        private BaseExperiment experimentCls;

        private Map<String, Object> experimentArgs;

        public ExperimentBean(String experimentName, float ratio, BaseExperiment experimentCls, Map<String, Object> experimentArgs) {
            this.experimentName = experimentName;
            this.ratio = ratio;
            this.experimentCls = experimentCls;
            this.experimentArgs = experimentArgs;
        }
    }

    private Map<String, Scene> initScenes(SceneConfig sceneConfig, ApplicationContext ctx) {
        HashMap<String, Scene> scenes = new HashMap<>();
        Map<String, Object> layerBeanObjects = ctx.getBeansWithAnnotation(LayerAnnotation.class);
        Map<String, Object> experimentBeanObjects = ctx.getBeansWithAnnotation(ExperimentAnnotation.class);
        for (com.dmetasoul.metaspore.pipeline.pojo.Scene scene : sceneConfig.getScenes()) {
            String sceneName = scene.getName();
            // layers
            SceneImpl sceneImplBean = new SceneImpl();
            sceneImplBean.setSceneArgs(scene.getSceneArgs());
            for (Layer layer : scene.getLayers()) {
                // add layer class
                String layerName = layer.getName();
                LayerBean layerBean = new LayerBean();
                Optional<Map.Entry<String, Object>> layerBeanObject = layerBeanObjects.entrySet().stream().filter(map -> ctx.findAnnotationOnBean(map.getKey(), LayerAnnotation.class).name().equals(layerName)).findFirst();
                if (layerBeanObject.isPresent()) {
                    String layerBeanName = layerBeanObject.get().getKey();
                    BaseLayer layerCls = (BaseLayer) layerBeanObjects.get(layerBeanName);
                    List<NormalLayerArgs> normalLayerArgsList = layer.getNormalLayerArgs();
                    Map<String, Object> extraLayerArgs = layer.getExtraLayerArgs();
                    LayerArgs layerArgs = new LayerArgs(normalLayerArgsList, extraLayerArgs);
                    layerBean.setLayerName(layerName);
                    layerBean.setLayerClass(layerCls);
                    layerBean.setLayerArgs(layerArgs);
                    // layer.intitialize()
                    layerCls.intitialize(layerArgs);
                    //
                    for (NormalLayerArgs normalLayerArgs : normalLayerArgsList) {
                        String experimentName = normalLayerArgs.getExperimentName();
                        float ratio = normalLayerArgs.getRatio();
                        // add experiment class
                        Optional<Map.Entry<String, Object>> experimentBeanObject = experimentBeanObjects.entrySet().stream().filter(map -> ctx.findAnnotationOnBean(map.getKey(), ExperimentAnnotation.class).name().equals(experimentName)).findFirst();
                        if (experimentBeanObject.isPresent()) {
                            String experimentBeanName = experimentBeanObject.get().getKey();
                            // TODO: 2022/3/25 增加 cloneFromClass
                            BaseExperiment experimentCls = (BaseExperiment) experimentBeanObjects.get(experimentBeanName);
                            // add experimentArgs
                            Optional<Experiment> experimentConfig = sceneConfig.getExperiments().stream().filter(x -> x.getLayerName().equals(layerName) && x.getExperimentName().equals(experimentName)).findFirst();
                            Map<String, Object> experimentArgs = experimentConfig.get().getExperimentArgs();
                            // experiment initialize()
                            experimentCls.initialize(experimentArgs);
                            layerBean.putExperiment(experimentName, ratio, experimentCls, experimentArgs);
                        }
                    }
                }
                sceneImplBean.add(layerBean);
            }
            scenes.put(sceneName, sceneImplBean);
        }
        return scenes;
    }

//    private BaseExperiment getBeanFromCtxWithClassName(String experimentClassName) {
//
//        BaseExperiment experimentCls = (BaseExperiment) ApplicationContextGetBeanHelper.getBeanFromClassName(experimentClassName);
//        return experimentCls;
//
//    }
//
//    private BaseExperiment getObjectFromReflectWithClassName(String experimentClassName) {
//
//        BaseExperiment experimentCls = (BaseExperiment) ApplicationContextGetBeanHelper.getBeanFromClassName(experimentClassName);
//        return experimentCls;
//
//    }


    // TODO: 2022/3/16 增加上下层 input/output 类型校验
    private void checkInOutClassType(Class<?> a, Class<?> b) {
        if (a.isInstance(b)) {
        }
    }

}