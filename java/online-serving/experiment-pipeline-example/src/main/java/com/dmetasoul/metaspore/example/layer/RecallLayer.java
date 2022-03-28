package com.dmetasoul.metaspore.example.layer;

import com.dmetasoul.metaspore.pipeline.BaseLayer;
import com.dmetasoul.metaspore.pipeline.annotation.LayerAnnotation;
import com.dmetasoul.metaspore.pipeline.pojo.Context;
import com.dmetasoul.metaspore.pipeline.pojo.LayerArgs;
import org.springframework.stereotype.Component;

@LayerAnnotation(name = "recall")
@Component
class RecallLayer implements BaseLayer<MyExperimentPojo> {

    private LayerArgs layerArgs;

    @Override
    public void intitialize(LayerArgs args) {
        System.out.println("recall Layer intitialize ...");
        System.out.println("args: " + args);
        this.layerArgs = args;
    }


    @Override
    public String split(Context ctx, MyExperimentPojo input) {
        String returnExp = "milvus";
        System.out.printf("layer split: %s, return exp: %s%n", this.getClass().getName(), returnExp);
        System.out.println("global sceneArgs: " + ctx.getSceneArgs());
        return returnExp;
    }


}