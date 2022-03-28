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

@LayerAnnotation(name = "match")
@Component
public class MatchLayer implements BaseLayer<RecommendResult> {
    private LayerBucketizer bucketizer;

    @Override
    public void intitialize(LayerArgs args) {
        System.out.println("match layer, args:" + args);
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
        String returnExp = bucketizer.toBucket(recommendResult.getRecommendContext());
        // TODO we should avoid to reference the experiment name explicitly
        System.out.printf("layer split: %s, return exp: %s%n", this.getClass().getName(), returnExp);
        return returnExp;
    }
}