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

package com.dmetasoul.metaspore.demo.multimodal.abtesting.layer.base;

import com.dmetasoul.metaspore.demo.multimodal.abtesting.layer.bucketizer.LayerBucketizer;
import com.dmetasoul.metaspore.demo.multimodal.abtesting.layer.bucketizer.RandomLayerBucketizer;
import com.dmetasoul.metaspore.demo.multimodal.abtesting.layer.bucketizer.SHA256LayerBucketizer;
import com.dmetasoul.metaspore.demo.multimodal.model.SearchResult;
import com.dmetasoul.metaspore.pipeline.BaseLayer;
import com.dmetasoul.metaspore.pipeline.annotation.LayerAnnotation;
import com.dmetasoul.metaspore.pipeline.impl.Context;
import com.dmetasoul.metaspore.pipeline.pojo.LayerArgs;
import org.springframework.stereotype.Component;

import java.util.Map;

@LayerAnnotation(name = "rank.base")
@Component
public class RankLayer implements BaseLayer<SearchResult> {
    private LayerBucketizer bucketizer;

    @Override
    public void intitialize(LayerArgs layerArgs) {
        System.out.println("Rank layer, args:" + layerArgs);
        Map<String, Object> extraArgs = layerArgs.getExtraLayerArgs();
        String bucketizer = (String) extraArgs.getOrDefault("bucketizer", "sha256");
        switch (bucketizer.toLowerCase()) {
            case "random":
                this.bucketizer = new RandomLayerBucketizer(layerArgs);
                break;
            default:
                this.bucketizer = new SHA256LayerBucketizer(layerArgs);
        }
    }

    @Override
    public String split(Context ctx, SearchResult in) {
        String returnExp = bucketizer.toBucket(in.getSearchContext());
        System.out.printf("Rank layer split: %s, return exp: %s%n", this.getClass().getName(), returnExp);
        return returnExp;
    }
}
