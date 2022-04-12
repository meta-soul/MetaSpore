package com.dmetasoul.metaspore.demo.multimodel.abtesting.experiment.match;

import com.dmetasoul.metaspore.demo.multimodel.model.SearchResult;
import com.dmetasoul.metaspore.pipeline.BaseExperiment;
import com.dmetasoul.metaspore.pipeline.annotation.ExperimentAnnotation;
import com.dmetasoul.metaspore.pipeline.impl.Context;
import org.springframework.stereotype.Component;

import java.util.Map;

@ExperimentAnnotation(name = "match.base")
@Component
public class MatchExperiment implements BaseExperiment<SearchResult, SearchResult> {
    @Override
    public void initialize(Map<String, Object> args) {
        System.out.println("match.base initialize... " + args);
    }

    @Override
    public SearchResult run(Context ctx, SearchResult in) {
        System.out.println("match.base experiment, match:" + in.getSearchQuery());
        return in;
    }
}
