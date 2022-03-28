package com.dmetasoul.metaspore.demo.movielens.abtesting.layer.bucketizer;

import com.google.common.collect.Maps;
import com.dmetasoul.metaspore.demo.movielens.model.RecommendContext;
import com.dmetasoul.metaspore.pipeline.pojo.LayerArgs;
import com.dmetasoul.metaspore.pipeline.pojo.NormalLayerArgs;

import java.util.List;
import java.util.Map;

public class RandomLayerBucketizer implements LayerBucketizer{
    protected Map<Integer, String> layerMap;

    protected AliasSampler sampler;

    public RandomLayerBucketizer(LayerArgs layerArgs) {
        System.out.println("RandomLayerBucketizer, args:" + layerArgs);
        List<NormalLayerArgs> normalLayerArgsList = layerArgs.getNormalLayerArgs();
        System.out.println("RandomLayerBucketizer, init layer map...");
        layerMap = Maps.newHashMap();
        double[] prob = new double[normalLayerArgsList.size()];
        for (int i = 0; i < normalLayerArgsList.size(); i++) {
            NormalLayerArgs args = normalLayerArgsList.get(i);
            String experimentName = args.getExperimentName();
            float ratio = args.getRatio();
            layerMap.put(i, experimentName);
            prob[i] = ratio;
        }
        System.out.println("RandomLayerBucketizer, init sampler...");
        prob = normalize(prob);
        sampler = new AliasSampler(prob);
    }

    @Override
    public String toBucket(RecommendContext context) {
        int bucketNo = sampler.nextInt() % layerMap.size();
        return layerMap.get(bucketNo);
    }
}
