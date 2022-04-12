package com.dmetasoul.metaspore.demo.multimodel.abtesting.experiment.qp;

import com.dmetasoul.metaspore.demo.multimodel.model.SearchContext;
import com.dmetasoul.metaspore.demo.multimodel.model.SearchResult;
import com.dmetasoul.metaspore.pipeline.BaseExperiment;
import com.dmetasoul.metaspore.pipeline.annotation.ExperimentAnnotation;
import com.dmetasoul.metaspore.pipeline.impl.Context;
import org.springframework.stereotype.Component;

import java.util.Map;

@ExperimentAnnotation(name = "qp.base")
@Component
public class QPExperiment implements BaseExperiment<SearchResult, SearchResult> {
    @Override
    public void initialize(Map<String, Object> args) {
        System.out.println("qp.base initialize... " + args);
    }

    @Override
    public SearchResult run(Context ctx, SearchResult in) {
        System.out.println("qp.base experiment, userModel:" + in.getSearchQuery());
        return in;
    }
}
