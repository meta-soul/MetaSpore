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
import com.dmetasoul.metaspore.demo.movielens.model.RecommendResult;
import com.dmetasoul.metaspore.pipeline.annotation.ExperimentAnnotation;
import com.dmetasoul.metaspore.pipeline.impl.Context;
import org.springframework.stereotype.Component;

import java.util.List;
import java.util.Map;

@ExperimentAnnotation(name = "diversify.MMR")
@Component

public class MMRDiversifyExperiment extends DiversifyExperiment {
    protected int lamada;

    public MMRDiversifyExperiment(DiversifierService diversifierService) {
        super(diversifierService);
    }

    @Override
    public void initialize(Map<String, Object> map) {
        super.initialize(map);
        recommendContext.setLamada(this.lamada);
        recommendContext.setDiversifierName("MMRDiersifier");
    }

    @Override
    public RecommendResult run(Context context, RecommendResult recommendResult) {
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
