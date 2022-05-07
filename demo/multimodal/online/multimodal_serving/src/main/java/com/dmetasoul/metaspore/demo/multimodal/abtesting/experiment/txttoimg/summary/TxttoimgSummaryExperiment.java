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

package com.dmetasoul.metaspore.demo.multimodal.abtesting.experiment.txttoimg.summary;

import com.dmetasoul.metaspore.demo.multimodal.domain.TxtToImgDemo;
import com.dmetasoul.metaspore.demo.multimodal.repository.TxtToImgDemoRepository;
import com.dmetasoul.metaspore.demo.multimodal.model.ItemModel;
import com.dmetasoul.metaspore.demo.multimodal.model.SearchContext;
import com.dmetasoul.metaspore.demo.multimodal.model.SearchResult;
import com.dmetasoul.metaspore.pipeline.BaseExperiment;
import com.dmetasoul.metaspore.pipeline.annotation.ExperimentAnnotation;
import com.dmetasoul.metaspore.pipeline.impl.Context;
import com.google.common.collect.Lists;
import org.springframework.stereotype.Component;

import java.util.*;
import java.util.stream.Collectors;

@ExperimentAnnotation(name = "summary.t2i.base")
@Component
public class TxttoimgSummaryExperiment implements BaseExperiment<SearchResult, SearchResult> {
    private List<String> summaryFields;

    private final TxtToImgDemoRepository txtToImgDemoRepository;

    public TxttoimgSummaryExperiment(TxtToImgDemoRepository txtToImgDemoRepository) {
        this.txtToImgDemoRepository = txtToImgDemoRepository;
    }

    @Override
    public void initialize(Map<String, Object> args) {
        summaryFields = Lists.newArrayList(((LinkedHashMap<String, String>) args.getOrDefault("summaryFields", new LinkedHashMap<String, String>())).values());
        System.out.println("[TxtToImg] summary.base initialize... " + args);
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
        Collection<TxtToImgDemo> txtToImgItems = txtToImgDemoRepository.findByQueryidIn(ids);
        Map<String, TxtToImgDemo> txtToImgItemMap = new HashMap<>();
        for (TxtToImgDemo x : txtToImgItems) {
            txtToImgItemMap.put(x.getQueryid(), x);
        }

        for (int qid=0; qid<rankingItemModels.size(); qid++) {
            rankingItemModels.get(qid).forEach(item -> {
                TxtToImgDemo txtToImgDemo = txtToImgItemMap.get(item.getId());
                mapRepositoryToItemModel(txtToImgDemo, item);
                item.setScore(item.getFinalRankingScore());  // copy ranking score as final score
            });
        }

        in.setSearchItemModels(rankingItemModels);

        System.out.println("[TxtToImg] summary.base experiment, Query:" + in.getSearchQuery());
        return in;
    }

    private void mapRepositoryToItemModel(TxtToImgDemo txtToImgDemo, ItemModel itemModel) {
        for (String field : summaryFields) {
            switch (field) {
                case "name":
                    itemModel.setSummary(field, txtToImgDemo.getName());
                    break;
                case "url":
                    itemModel.setSummary(field, txtToImgDemo.getUrl());
                    break;
            }
        }
    }
}
