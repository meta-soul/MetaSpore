package com.dmetasoul.metaspore.demo.movielens.abtesting.layer;

import com.dmetasoul.metaspore.demo.movielens.model.RecommendContext;
import com.dmetasoul.metaspore.pipeline.BaseLayer;
import com.dmetasoul.metaspore.pipeline.annotation.LayerAnnotation;
import com.dmetasoul.metaspore.pipeline.pojo.Context;
import com.dmetasoul.metaspore.pipeline.pojo.LayerArgs;
import org.springframework.stereotype.Component;

@LayerAnnotation(name = "userModel")
@Component
public class UserModelLayer implements BaseLayer<RecommendContext> {

    @Override
    public void intitialize(LayerArgs args) {
        System.out.println("userModel layer, args:" + args);
    }

    @Override
    public String split(Context context, RecommendContext recommendContext) {
        String returnExp = "userModel.base";
        // TODO we should avoid to reference the experiment name explicitly
        System.out.printf("layer split: %s, return exp: %s%n", this.getClass().getName(), returnExp);
        return returnExp;
    }
}
