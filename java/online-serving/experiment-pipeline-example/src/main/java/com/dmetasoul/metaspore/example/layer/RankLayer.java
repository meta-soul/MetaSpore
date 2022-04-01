package com.dmetasoul.metaspore.example.layer;

import com.dmetasoul.metaspore.pipeline.BaseLayer;
import com.dmetasoul.metaspore.pipeline.annotation.LayerAnnotation;
import com.dmetasoul.metaspore.pipeline.impl.Context;
import com.dmetasoul.metaspore.pipeline.pojo.LayerArgs;
import org.springframework.stereotype.Component;

@LayerAnnotation(name = "rank")
@Component
class RankLayer implements BaseLayer<SecondLayerPojo> {
    private LayerArgs layerArgs;

    @Override
    public void intitialize(LayerArgs args) {
        System.out.println("rank Layer intitialize ...");
        System.out.println("args: " + args);
        this.layerArgs = args;
    }


    @Override
    public String split(Context ctx, SecondLayerPojo input) {
        String returnExp = "RankExperimentTwo";
        System.out.printf("layer: %s, split return exp: %s%n", this.getClass().getName(), returnExp);
        System.out.println("layerArgs: " + layerArgs);
        return returnExp;
    }
}