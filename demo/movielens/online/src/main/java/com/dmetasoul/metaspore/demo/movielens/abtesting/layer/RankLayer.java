package com.dmetasoul.metaspore.demo.movielens.abtesting.layer;

import com.dmetasoul.metaspore.demo.movielens.abtesting.layer.bucketizer.LayerBucketizer;
import com.dmetasoul.metaspore.demo.movielens.abtesting.layer.bucketizer.RandomLayerBucketizer;
import com.dmetasoul.metaspore.demo.movielens.abtesting.layer.bucketizer.SHA256LayerBucketizer;
import com.dmetasoul.metaspore.demo.movielens.model.RecommendResult;
import com.dmetasoul.metaspore.pipeline.BaseLayer;
import com.dmetasoul.metaspore.pipeline.annotation.LayerAnnotation;
import com.dmetasoul.metaspore.pipeline.pojo.Context;
import com.dmetasoul.metaspore.pipeline.pojo.LayerArgs;
import org.springframework.stereotype.Component;

import java.util.Map;

@LayerAnnotation(name = "rank")
@Component
public class RankLayer implements BaseLayer<RecommendResult> {
    private LayerBucketizer bucketizer;

    @Override
    public void intitialize(LayerArgs args) {
        System.out.println("rank layer, args:" + args);
        Map<String, Object> extraArgs = args.getExtraLayerArgs();
        String bucketizerConfig = (String) extraArgs.getOrDefault("bucketizer", "sha256");
        switch (bucketizerConfig.toLowerCase()) {
            case "random":
                this.bucketizer = new RandomLayerBucketizer(args);
                break;
            default:
                this.bucketizer = new SHA256LayerBucketizer(args);
        }
    }

    @Override
    public String split(Context context, RecommendResult recommendResult) {
        // TODO we should avoid to reference the experiment name explicitly
        String returnExp = bucketizer.toBucket(recommendResult.getRecommendContext());
        System.out.printf("layer split: %s, return exp: %s%n", this.getClass().getName(), returnExp);
        return returnExp;
    }
}
