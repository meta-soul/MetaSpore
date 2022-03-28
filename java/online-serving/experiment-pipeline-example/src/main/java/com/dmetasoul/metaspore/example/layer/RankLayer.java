package com.dmetasoul.metaspore.example.layer;

import com.dmetasoul.metaspore.pipeline.BaseLayer;
import com.dmetasoul.metaspore.pipeline.annotation.LayerAnnotation;
import com.dmetasoul.metaspore.pipeline.pojo.Context;
import com.dmetasoul.metaspore.pipeline.pojo.LayerArgs;
import org.springframework.stereotype.Component;

@LayerAnnotation(name = "rank")
@Component
class RankLayer implements BaseLayer<MyExperimentPojo2> {

    @Override
    public void intitialize(LayerArgs args) {
        System.out.println("rank Layer intitialize ...");
        System.out.println("args: " + args);
    }


    @Override
    public String split(Context ctx, MyExperimentPojo2 input) {
        String returnExp = "milvus3";
        System.out.printf("layer split: %s, return exp: %s%n", this.getClass().getName(), returnExp);
        return returnExp;
    }


}