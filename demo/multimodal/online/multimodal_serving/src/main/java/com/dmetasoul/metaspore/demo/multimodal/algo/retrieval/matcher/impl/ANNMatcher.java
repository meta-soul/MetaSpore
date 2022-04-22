package com.dmetasoul.metaspore.demo.multimodal.algo.retrieval.matcher.impl;

import com.dmetasoul.metaspore.demo.multimodal.algo.retrieval.matcher.Matcher;
import com.dmetasoul.metaspore.demo.multimodal.model.ItemModel;
import com.dmetasoul.metaspore.demo.multimodal.model.QueryModel;
import com.dmetasoul.metaspore.demo.multimodal.model.SearchContext;
import com.dmetasoul.metaspore.demo.multimodal.service.NpsService;
import com.dmetasoul.metaspore.demo.multimodal.service.MilvusService;
import com.dmetasoul.metaspore.serving.ArrowTensor;
import com.google.protobuf.ByteString;
import io.milvus.response.SearchResultsWrapper;
import lombok.SneakyThrows;
import org.hibernate.cache.spi.support.AbstractReadWriteAccess;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;

@Service
public class ANNMatcher implements Matcher {
    public static final String ALGO_NAME = "ann_matcher";
    public static final int DEFAULT_ALGO_LEVEL = 3;

    private NpsService npsService;
    private MilvusService milvusService;

    public ANNMatcher(NpsService npsService, MilvusService milvusService) {
        this.npsService = npsService;
        this.milvusService = milvusService;
    }

    @SneakyThrows
    @Override
    public List<List<ItemModel>> match(SearchContext searchContext, QueryModel queryModel) {
        // query embedding
        String modelName = searchContext.getMatchEmbeddingModelName();
        String vectorName = searchContext.getMatchEmbeddingVectorName();
        Map<String, ByteString> qpResults = searchContext.getQpResults();
        Map<String, ArrowTensor> npsResults = npsService.predictBlocking(modelName, qpResults, Collections.emptyMap());
        List<List<Float>> queryEmbs = npsService.getFloatVectorsFromNpsResult(npsResults, vectorName);

        // ann retrieval doc collections
        milvusService.setMilvusArgs(searchContext.getMatchMilvusArgs());
        Integer maxReservation = searchContext.getMatchMaxReservation();
        Map<Integer, List<SearchResultsWrapper.IDScore>> idScores = milvusService.findByEmbeddingVectors(queryEmbs, maxReservation);

        // make itemModels to return
        List<List<ItemModel>> allItemModels = new ArrayList<>();
        for (Integer qid : idScores.keySet()) {
            List<ItemModel> itemModels = new ArrayList<>();
            List<SearchResultsWrapper.IDScore> oneIdScores = idScores.get(qid);

            // desc order
            oneIdScores.sort((o1, o2) -> Double.compare(o2.getScore(), o1.getScore()));

            Double maxScore = oneIdScores.size() > 0 ? oneIdScores.get(0).getScore() : 0.0;
            for (int i=0; i < oneIdScores.size(); i++) {
                SearchResultsWrapper.IDScore x = oneIdScores.get(i);
                Double score = (double) x.getScore();
                ItemModel item = new ItemModel();
                // take Milvus id as Item's id
                item.setId(String.valueOf(x.getLongID()));
                // save score of corresponding retrieval algo
                item.setOriginalRetrievalScore(ALGO_NAME, score);
                // final score is normalized
                item.setFinalRetrievalScore(Matcher.getFinalRetrievalScore(score, maxScore, DEFAULT_ALGO_LEVEL));
                itemModels.add(item);
            }
            allItemModels.add(itemModels);
        }

        return allItemModels;
    }
}
