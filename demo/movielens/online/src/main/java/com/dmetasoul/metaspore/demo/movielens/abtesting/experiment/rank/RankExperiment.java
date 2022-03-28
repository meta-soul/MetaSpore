package com.dmetasoul.metaspore.demo.movielens.abtesting.experiment.rank;

import com.dmetasoul.metaspore.demo.movielens.model.RecommendResult;
import com.dmetasoul.metaspore.demo.movielens.ranking.RankingService;
import com.dmetasoul.metaspore.demo.movielens.ranking.ranker.RankingSortStrategy;
import com.dmetasoul.metaspore.pipeline.BaseExperiment;
import com.dmetasoul.metaspore.pipeline.annotation.ExperimentAnnotation;
import com.dmetasoul.metaspore.pipeline.pojo.Context;
import lombok.SneakyThrows;
import org.springframework.stereotype.Component;

import java.util.Map;

@ExperimentAnnotation(name = "rank.base")
@Component
public class RankExperiment implements BaseExperiment<RecommendResult, RecommendResult>  {
    protected final RankingService rankingService;

    protected String rankerName;
    protected RankingSortStrategy.Type sortStrategyType;
    protected Double sortStrategyAlpha;
    protected Double sortStrategyBeta;
    protected Integer maxReservation;
    protected String modelName;

    public RankExperiment(RankingService rankingService) {
        this.rankingService = rankingService;
    }

    @Override
    public void initialize(Map<String, Object> map) {
        // TODO make sure rankerNames and maxReservation can be updated safely here in multi-thread scenarios
        this.rankerName = (String) map.get("ranker");

        // TODO consider passing the raw Strategy string instead of Strategy enum.
        try {
            sortStrategyType =  RankingSortStrategy.Type.valueOf((String) map.get("sortStrategyType"));
        } catch (Exception e) {
            // TODO add warn log.
            sortStrategyType = RankingSortStrategy.DEFAULT_STRATEGY;
        }

        this.sortStrategyAlpha = (Double) map.get("sortStrategyAlpha");
        this.sortStrategyBeta = (Double) map.get("sortStrategyBeta");
        this.maxReservation = (Integer) map.get("maxReservation");
        this.modelName = (String) map.get("modelName");
    }

    @SneakyThrows
    @Override
    public RecommendResult run(Context context, RecommendResult recommendResult) {
        System.out.println("rank.base experiment, userModel:" + recommendResult.getUserId());
        System.out.println("rank.base experiment, return the upper layer result directly.");
        return recommendResult;
    }
}
