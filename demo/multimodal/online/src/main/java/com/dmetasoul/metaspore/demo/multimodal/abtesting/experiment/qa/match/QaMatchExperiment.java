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

package com.dmetasoul.metaspore.demo.multimodal.abtesting.experiment.qa.match;

import com.dmetasoul.metaspore.demo.multimodal.model.ItemModel;
import com.dmetasoul.metaspore.demo.multimodal.model.SearchResult;
import com.dmetasoul.metaspore.demo.multimodal.model.SearchContext;
import com.dmetasoul.metaspore.pipeline.BaseExperiment;
import com.dmetasoul.metaspore.pipeline.annotation.ExperimentAnnotation;
import com.dmetasoul.metaspore.pipeline.impl.Context;
import lombok.SneakyThrows;
import org.springframework.stereotype.Component;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.Map;

@ExperimentAnnotation(name = "match.qa.base")
@Component
public class QaMatchExperiment implements BaseExperiment<SearchResult, SearchResult> {
    private String modelName;

    @Override
    public void initialize(Map<String, Object> args) {
        modelName = (String) args.get("modelName");

        System.out.println("match.base initialize... " + args);
    }

    @SneakyThrows
    @Override
    public SearchResult run(Context ctx, SearchResult in) {
        SearchContext searchContext = in.getSearchContext();
        searchContext.setMatchEmbeddingModelName(modelName);

        //List<ItemModel> itemModels = retrievalService.match(in.getSearchContext(), in.getQueryModel());

        System.out.println("match.base experiment, Query:" + in.getSearchQuery());
        return in;
    }
}
