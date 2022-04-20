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

package com.dmetasoul.metaspore.demo.multimodal.abtesting.experiment.qa.rank;


import com.dmetasoul.metaspore.demo.multimodal.model.ItemModel;
import com.dmetasoul.metaspore.demo.multimodal.model.SearchContext;
import com.dmetasoul.metaspore.demo.multimodal.model.SearchResult;
import com.dmetasoul.metaspore.pipeline.BaseExperiment;
import com.dmetasoul.metaspore.pipeline.annotation.ExperimentAnnotation;
import com.dmetasoul.metaspore.pipeline.impl.Context;
import org.springframework.stereotype.Component;

import java.util.List;
import java.util.Map;

@ExperimentAnnotation(name = "rank.qa.base")
@Component
public class QaRankExperiment implements BaseExperiment<SearchResult, SearchResult> {
    @Override
    public void initialize(Map<String, Object> args) {
        System.out.println("rank.base initialize... " + args);
    }

    @Override
    public SearchResult run(Context ctx, SearchResult in) {
        System.out.println("rank layer do nothing, just return for now!");

        // just copy match results
        SearchContext searchContext = in.getSearchContext();
        List<List<ItemModel>> itemModels = searchContext.getMatchItemModels();
        searchContext.setRankItemModels(itemModels);  // set for downstream pipeline
        in.setSearchItemModels(itemModels);  // the final search results set for now

        //System.out.println(itemModels);
        System.out.println("rank.base experiment, Query:" + in.getSearchQuery());
        return in;
    }
}
