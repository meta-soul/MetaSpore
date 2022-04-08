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

package com.dmetasoul.metaspore.demo.movielens.abtesting.experiment.diversify;

import com.dmetasoul.metaspore.demo.movielens.diversify.DiversifierService;
import com.dmetasoul.metaspore.demo.movielens.model.ItemModel;
import com.dmetasoul.metaspore.demo.movielens.model.RecommendContext;
import com.dmetasoul.metaspore.demo.movielens.model.RecommendResult;
import com.dmetasoul.metaspore.pipeline.BaseExperiment;
import com.dmetasoul.metaspore.pipeline.annotation.ExperimentAnnotation;
//import com.dmetasoul.metaspore.pipeline.impl.Context;
import com.dmetasoul.metaspore.pipeline.pojo.Context;
import org.springframework.stereotype.Component;

import java.util.List;
import java.util.Map;

@ExperimentAnnotation(name = "diversify.base")
@Component

public class DiversifyExperiment implements BaseExperiment<RecommendResult, RecommendResult> {
    protected final DiversifierService diversifierService;

    protected boolean useDiversify = true;

    protected int window;

    protected int tolerance;

    protected String diverdifierName;

    //protected RecommendContext recommendContext=new RecommendContext();

    public DiversifyExperiment(DiversifierService diversifierService) {
        this.diversifierService = diversifierService;
    }

    @Override
    public void initialize(Map<String, Object> map) {
        this.useDiversify = (Boolean) map.getOrDefault("useDiversify", Boolean.TRUE);
        this.window = (int) map.getOrDefault("window",4);
        this.tolerance = (int) map.getOrDefault("tolerance",4);
        this.diverdifierName=(String)map.getOrDefault("diverisifier", "SimpleDiversifier");
        //this.recommendContext = recommendContext;
        //recommendContext.setDiversifierName((String) map.getOrDefault("diverisifier", "SimpleDiversifier"));
        System.out.println("diversify.base initialize, useDiversify:" + this.useDiversify
                + ", window:" + this.window
                + ", tolerance:" + this.tolerance
                + ",diverdifierMethod:" + this.diverdifierName);
    }

    @Override
    public RecommendResult run(Context context, RecommendResult recommendResult) {
        RecommendContext recommendContext = recommendResult.getRecommendContext();
        recommendContext.setDiversifierName(this.diverdifierName );
        List<ItemModel> itemModel = recommendResult.getRecommendItemModels();

        if (!useDiversify) {
            System.out.println("diversify.base experiment, turn off diversify");
            return recommendResult;
        }
        List<ItemModel> diverseItemModels = diversifierService.diverse(recommendContext, itemModel, this.window, this.tolerance);

        recommendResult.setRecommendItemModels(diverseItemModels);
        return recommendResult;
    }
}