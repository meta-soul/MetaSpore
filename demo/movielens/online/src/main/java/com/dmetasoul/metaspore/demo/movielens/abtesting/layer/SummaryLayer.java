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