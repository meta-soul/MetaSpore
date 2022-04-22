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

package com.dmetasoul.metaspore.demo.multimodal.abtesting.experiment.qa.summary;

import com.dmetasoul.metaspore.demo.multimodal.model.ItemModel;
import com.dmetasoul.metaspore.demo.multimodal.model.SearchContext;
import com.dmetasoul.metaspore.demo.multimodal.model.SearchResult;
import com.dmetasoul.metaspore.demo.multimodal.domain.BaikeQaDemo;
import com.dmetasoul.metaspore.demo.multimodal.repository.BaikeQaDemoRepository;
import com.dmetasoul.metaspore.pipeline.BaseExperiment;
import com.dmetasoul.metaspore.pipeline.annotation.ExperimentAnnotation;
import com.dmetasoul.metaspore.pipeline.impl.Context;
import com.google.common.collect.Lists;
import org.springframework.stereotype.Component;

import java.util.*;
import java.util.stream.Collectors;

@ExperimentAnnotation(name = "summary.qa.base")
@Component
public class QaSummaryExperiment implements BaseExperiment<SearchResult, SearchResult> {
    private List<String> summaryFields;

    private final BaikeQaDemoRepository baikeQaDemoRepository;

    public QaSummaryExperiment(BaikeQaDemoRepository baikeQaDemoRepository) { this.baikeQaDemoRepository = baikeQaDemoRepository; }

    @Override
    public void initialize(Map<String, Object> args) {
        summaryFields = Lists.newArrayList(((LinkedHashMap<String, String>) args.getOrDefault("summaryFields", new LinkedHashMap<String, String>())).values());
        System.out.println("summary.base initialize... " + args);
    }

    @Override
    public SearchResult run(Context ctx, SearchResult in) {
        SearchContext searchContext = in.getSearchContext();
        List<List<ItemModel>> rankingItemModels = in.getSearchItemModels();

        if (rankingItemModels.size() == 0) {
            System.out.println("Ranking results is empty!!!");
            return in;
        }

        List<String> ids = new ArrayList<>();
        rankingItemModels.forEach(x -> {
            ids.addAll(x.stream().map(ItemModel::getId).collect(Collectors.toList()));
        });
        Collection<BaikeQaDemo> baikeQaItems = baikeQaDemoRepository.findByQueryidIn(ids);
        Map<String, BaikeQaDemo> baikeQaItemMap = new HashMap<>();
        for (BaikeQaDemo x : baikeQaItems) {
            baikeQaItemMap.put(x.getQueryid(), x);
        }

        for (int qid=0; qid<rankingItemModels.size(); qid++) {
            rankingItemModels.get(qid).forEach(item -> {
                BaikeQaDemo baikeQaDemo = baikeQaItemMap.get(item.getId());
                mapRepositoryToItemModel(baikeQaDemo, item);
                item.setScore(item.getFinalRankingScore());  // copy ranking score as final score
            });
        }

        in.setSearchItemModels(rankingItemModels);

        System.out.println("summary.base experiment, Query:" + in.getSearchQuery());
        return in;
    }

    private void mapRepositoryToItemModel(BaikeQaDemo baikeQaDemo, ItemModel itemModel) {
        for (String field : summaryFields) {
            switch (field) {
                case "question":
                    itemModel.setSummary(field, baikeQaDemo.getQuestion());
                    break;
                case "answer":
                    itemModel.setSummary(field, baikeQaDemo.getAnswer());
                    break;
                case "category":
                    itemModel.setSummary(field, baikeQaDemo.getCategory());
                    break;
            }
        }
    }
}
