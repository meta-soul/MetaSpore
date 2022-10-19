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
import com.dmetasoul.metaspore.pipeline.pojo.LayerContext;
import lombok.Data;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

@Data
public class SceneImpl implements Scene {
    private List<ScenesFactoryImpl.LayerBean> layers = new ArrayList<>();
    private Map<String, Object> extraSceneArgs = new HashMap<>();

    @Override
    public Object run(Object in) {

        Context ctx = new Context();
        return run(in, ctx);
    }

    @Override
    public Object run(Object in, Context ctx) {

        return execute(in, false, ctx, null);
    }

    @Override
    public Object runDebug(Object in, Map<String, String> specifiedLayerAndExperiment) {

        Context ctx = new Context();
        return runDebug(in, ctx, specifiedLayerAndExperiment);
    }

    @Override
    public Object runDebug(Object in, Context ctx, Map<String, String> specifiedLayerAndExperiment) {
        return execute(in, true, ctx, specifiedLayerAndExperiment);
    }

    public void add(ScenesFactoryImpl.LayerBean layerBean) {
        layers.add(layerBean);
    }

    private Object execute(Object in, Boolean isDebugMode, Context ctx, Map<String, String> specifiedLayerAndExperiment) {
//        ctx.setExtraSceneArgs(extraSceneArgs);
        for (int i = 0; i < layers.size(); i++) {
            ScenesFactoryImpl.LayerBean layer = layers.get(i);
            LayerContext layerContext = new LayerContext();
            String layerName = layer.getLayerName();
            if (i == 0) {
                layerContext.setInput(in);
            } else {
                Object lastOutput = ctx.getLayerContexts().get(i - 1).getOutput();
                layerContext.setInput(lastOutput);
            }

            layerContext.setLayerNum(i);
            layerContext.setLayerName(layerName);
            BaseLayer layerClass = layer.getLayerBeanObject();

            // layer.split()
            String splitExperimentName = layerClass.split(ctx, layerContext.getInput());
            // debug mode
            if (isDebugMode) {
                splitExperimentName = getDebugExperiment(layerName, specifiedLayerAndExperiment, splitExperimentName);
            }

            ScenesFactoryImpl.ExperimentBean splitExperimentBean = layer.getExperimentBean(splitExperimentName);
            BaseExperiment experimentObject = splitExperimentBean.getExperimentObject();

            // experiment run()
            Object ouput = experimentObject.run(ctx, layerContext.getInput());
            layerContext.setSplitExperimentName(splitExperimentName);
            layerContext.setOutput(ouput);
            ctx.addLayerContext(layerContext);
        }
        return ctx.getLayerContexts().get(ctx.getLayerContexts().size() - 1).getOutput();
    }

    private String getDebugExperiment(String layerName, Map<String, String> specifiedLayerAndExperiment, String defaultExperiment) {
        if (specifiedLayerAndExperiment.keySet().contains(layerName)) {
            return specifiedLayerAndExperiment.get(layerName);
        } else {
            return defaultExperiment;
        }
    }
}