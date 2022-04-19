package com.dmetasoul.metaspore.demo.multimodal.algo.qp;

import com.dmetasoul.metaspore.demo.multimodal.model.QueryModel;
import com.dmetasoul.metaspore.demo.multimodal.model.SearchContext;
import java.io.IOException;

public interface QpService {
    void process(SearchContext searchContext, QueryModel queryModel) throws IOException;
}
