package com.dmetasoul.metaspore.example.layer;

import com.dmetasoul.metaspore.pipeline.BaseLayer;
import com.dmetasoul.metaspore.pipeline.annotation.LayerAnnotation;
import com.dmetasoul.metaspore.pipeline.impl.Context;
import com.dmetasoul.metaspore.pipeline.pojo.LayerArgs;
import org.springframework.stereotype.Component;

@LayerAnnotation(name = "recall")
@Component
class RecallLayer implements BaseLayer<FirstLayerPojo> {

    private LayerArgs layerArgs;

    @Override
    public void intitialize(LayerArgs args) {
        System.out.println("recall Layer intitialize ...");
        System.out.println("args: " + args);
        this.layerArgs = args;
    }


    @Override
    public String split(Context ctx, FirstLayerPojo input) {
        String returnExp = "RecallExperimentOne";
        System.out.printf("layer: %s, split return exp: %s%n", this.getClass().getName(), returnExp);
        System.out.println("layerArgs: " + layerArgs);
        return returnExp;
    }


}