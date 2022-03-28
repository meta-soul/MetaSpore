package com.dmetasoul.metaspore.demo.movielens.abtesting.layer.bucketizer;

import com.google.common.base.Charsets;
import com.google.common.collect.Maps;
import com.google.common.hash.HashCode;
import com.google.common.hash.Hasher;
import com.google.common.hash.Hashing;
import com.dmetasoul.metaspore.demo.movielens.model.RecommendContext;
import com.dmetasoul.metaspore.pipeline.pojo.LayerArgs;
import com.dmetasoul.metaspore.pipeline.pojo.NormalLayerArgs;

import java.util.List;
import java.util.Map;

// References:
// * https://mojito.mx/docs/example-hash-function-split-test-assignment
// * https://engineering.depop.com/a-b-test-bucketing-using-hashing-475c4ce5d07

public class SHA256LayerBucketizer implements LayerBucketizer {
    protected ArraySampler sampler;

    protected String salt = "i like movie lens project";

    protected Map<Integer, String> layerMap;

    public SHA256LayerBucketizer(LayerArgs layerArgs) {
        System.out.println("SHA256LayerBucketizer, args:" + layerArgs);
        List<NormalLayerArgs> normalLayerArgsList = layerArgs.getNormalLayerArgs();
        System.out.println("SHA256LayerBucketizer, init layer map...");
        layerMap = Maps.newHashMap();
        double[] prob = new double[normalLayerArgsList.size()];
        for (int i = 0; i < normalLayerArgsList.size(); i++) {
            NormalLayerArgs args = normalLayerArgsList.get(i);
            String experimentName = args.getExperimentName();
            float ratio = args.getRatio();
            layerMap.put(i, experimentName);
            prob[i] = ratio;
        }
        System.out.println("RandomLayer layer, init sampler...");
        prob = normalize(prob);
        sampler = new ArraySampler(prob);
    }

    @Override
    public String toBucket(RecommendContext context) {
        HashCode sha256 = sha256(context.getUserId());
        int bucketNo = sampler.nextInt(sha256) % layerMap.size();
        return layerMap.get(bucketNo);
    }

    protected HashCode sha256(String userId) {
        String combination = userId + "#" + salt;
        Hasher hasher = Hashing.sha256().newHasher();
        hasher.putString(combination, Charsets.UTF_8);
        HashCode sha256 = hasher.hash();
        return sha256;
    }
}
