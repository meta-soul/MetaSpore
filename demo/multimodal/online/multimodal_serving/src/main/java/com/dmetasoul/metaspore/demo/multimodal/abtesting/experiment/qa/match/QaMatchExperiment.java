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
import com.dmetasoul.metaspore.demo.multimodal.algo.retrieval.RetrievalService;
import com.dmetasoul.metaspore.pipeline.BaseExperiment;
import com.dmetasoul.metaspore.pipeline.annotation.ExperimentAnnotation;
import com.dmetasoul.metaspore.pipeline.impl.Context;
import lombok.SneakyThrows;
import org.springframework.stereotype.Component;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.LinkedHashMap;
import java.util.stream.Collectors;

import com.google.common.collect.Lists;

@ExperimentAnnotation(name = "match.qa.base")
@Component
public class QaMatchExperiment implements BaseExperiment<SearchResult, SearchResult> {
    private String modelName;
    private String vectorName;
    private List<String> matcherNames;
    private Integer maxReservation;
    private Map<String, String> milvusArgs;
    protected final RetrievalService retrievalService;

    public QaMatchExperiment(RetrievalService retrievalService) {
        this.retrievalService = retrievalService;
        this.milvusArgs = new HashMap<>();
        this.matcherNames = new ArrayList<>();
    }

    @Override
    public void initialize(Map<String, Object> args) {
        modelName = (String) args.get("modelName");
        vectorName = (String) args.get("vectorName");
        //matcherNames = (List<String>) args.get("matcherNames");
        matcherNames = Lists.newArrayList(((LinkedHashMap<String, String>) args.getOrDefault("matcherNames", new LinkedHashMap<String, String>())).values());
        maxReservation = (Integer) args.get("maxReservation");
        this.milvusArgs.putAll((Map<String, String>) args.get("milvusArgs"));
        System.out.println("match.base initialize... " + args);
    }

    @SneakyThrows
    @Override
    public SearchResult run(Context ctx, SearchResult in) {
        SearchContext searchContext = in.getSearchContext();
        searchContext.setMatchEmbeddingModelName(modelName);
        searchContext.setMatchEmbeddingVectorName(vectorName);
        searchContext.setMatchMatcherNames(matcherNames);
        searchContext.setMatchMaxReservation(maxReservation);
        searchContext.setMatchMilvusArgs(milvusArgs);

        List<List<ItemModel>> itemModels = retrievalService.match(in.getSearchContext(), in.getQueryModel());
        searchContext.setMatchItemModels(itemModels);  // set for downstream pipeline
        in.setSearchItemModels(itemModels);  // the final search results set for now

        System.out.println("match.base experiment, Query:" + in.getSearchQuery() + ", Items:" + String.valueOf(itemModels.stream().map(List::size).collect(Collectors.toList())));
        return in;
    }
}
