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

package com.dmetasoul.metaspore.demo.multimodal.abtesting.experiment.txttoimg.rank;


import com.dmetasoul.metaspore.demo.multimodal.model.ItemModel;
import com.dmetasoul.metaspore.demo.multimodal.model.SearchContext;
import com.dmetasoul.metaspore.demo.multimodal.model.SearchResult;
import com.dmetasoul.metaspore.pipeline.BaseExperiment;
import com.dmetasoul.metaspore.pipeline.annotation.ExperimentAnnotation;
import com.dmetasoul.metaspore.pipeline.impl.Context;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

@ExperimentAnnotation(name = "rank.t2i.base")
@Component
public class TxttoimgRankExperiment implements BaseExperiment<SearchResult, SearchResult> {
    private Integer maxReservation;

    @Override
    public void initialize(Map<String, Object> args) {
        maxReservation = (Integer) args.get("maxReservation");
        System.out.println("[TxtToImg] rank.base initialize... " + args);
    }

    @Override
    public SearchResult run(Context ctx, SearchResult in) {
        SearchContext searchContext = in.getSearchContext();
        searchContext.setRankMaxReservation(maxReservation);

        // just copy match results
        System.out.println("rank layer do nothing, just return for now!");
        Integer maxReservation = searchContext.getRankMaxReservation();
        List<List<ItemModel>> itemModels = searchContext.getMatchItemModels();
        List<List<ItemModel>> rankingItemModels = new ArrayList<>();
        for (int qid=0; qid<itemModels.size(); qid++) {
            itemModels.set(qid, itemModels.get(qid).subList(0, maxReservation));  // truncate
            itemModels.get(qid).forEach(x -> {
                x.setOriginalRankingScoreMap("dummy", x.getFinalRetrievalScore());
                x.setFinalRankingScore(x.getFinalRetrievalScore());
            });
            rankingItemModels.add(itemModels.get(qid));
        }

        searchContext.setRankItemModels(rankingItemModels);  // set for downstream pipeline
        in.setSearchItemModels(rankingItemModels);  // the final search results set for now

        //System.out.println(itemModels);
        System.out.println("[TxtToImg] rank.base experiment, Query:" + in.getSearchQuery() + ", Items:" + String.valueOf(rankingItemModels.stream().map(List::size).collect(Collectors.toList())));

        return in;
    }
}
