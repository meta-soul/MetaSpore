package com.dmetasoul.metaspore.demo.multimodal.algo.qp.processor;

import com.dmetasoul.metaspore.demo.multimodal.model.QueryModel;
import com.dmetasoul.metaspore.demo.multimodal.model.SearchContext;
import java.io.IOException;

public interface Processor {
    void process(SearchContext searchContext, QueryModel queryModel) throws IOException;
}
