package com.dmetasoul.metaspore.demo.movielens.abtesting.layer;

import com.dmetasoul.metaspore.demo.movielens.model.RecommendResult;
import com.dmetasoul.metaspore.pipeline.BaseLayer;
import com.dmetasoul.metaspore.pipeline.annotation.LayerAnnotation;
import com.dmetasoul.metaspore.pipeline.pojo.Context;
import com.dmetasoul.metaspore.pipeline.pojo.LayerArgs;
import org.springframework.stereotype.Component;

@LayerAnnotation(name = "summary")
@Component
public class SummaryLayer implements BaseLayer<RecommendResult> {
    @Override
    public void intitialize(LayerArgs args) {
        System.out.println("summary layer, args:" + args);
    }

    @Override
    public String split(Context context, RecommendResult recommendResult) {
        // TODO we should avoid to reference the experiment name explicitly
        String returnExp = "summary.base";
        System.out.printf("layer split: %s, return exp: %s%n", this.getClass().getName(), returnExp);
        return returnExp;
    }
}
