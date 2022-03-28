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