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

package com.dmetasoul.metaspore.demo.multimodal.abtesting.layer.bucketizer;

import com.dmetasoul.metaspore.demo.multimodal.model.SearchContext;
import com.dmetasoul.metaspore.pipeline.pojo.LayerArgs;
import com.dmetasoul.metaspore.pipeline.pojo.NormalLayerArgs;
import com.google.common.base.Charsets;
import com.google.common.collect.Maps;
import com.google.common.hash.HashCode;
import com.google.common.hash.Hasher;
import com.google.common.hash.Hashing;

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
    public String toBucket(SearchContext context) {
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