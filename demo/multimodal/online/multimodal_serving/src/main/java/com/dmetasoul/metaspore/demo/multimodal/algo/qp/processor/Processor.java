package com.dmetasoul.metaspore.demo.multimodal.algo.qp.processor;

import com.dmetasoul.metaspore.demo.multimodal.model.QueryModel;
import com.dmetasoul.metaspore.demo.multimodal.model.SearchContext;
import com.google.protobuf.ByteString;

import java.io.IOException;
import java.util.Map;

public interface Processor {
    Map<String, ByteString> process(SearchContext searchContext, QueryModel queryModel) throws IOException;
}
