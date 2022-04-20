package com.dmetasoul.metaspore.demo.multimodal.algo.qp.impl;

import com.dmetasoul.metaspore.demo.multimodal.algo.qp.QpService;
import com.dmetasoul.metaspore.demo.multimodal.algo.qp.processor.Processor;
import com.dmetasoul.metaspore.demo.multimodal.algo.qp.processor.ProcessorProvider;
import com.dmetasoul.metaspore.demo.multimodal.model.QueryModel;
import com.dmetasoul.metaspore.demo.multimodal.model.SearchContext;
import com.google.protobuf.ByteString;
import org.springframework.stereotype.Service;

import java.io.IOException;
import java.util.Map;

@Service
public class QpServiceImpl implements QpService {

    private final ProcessorProvider processorProvider;

    public QpServiceImpl(ProcessorProvider processorProvider) {
        this.processorProvider = processorProvider;
    }

    @Override
    public Map<String, ByteString> process(SearchContext searchContext, QueryModel queryModel) throws IOException {
        String processorName = searchContext.getQpQueryProcessorModelName();
        Processor processor = processorProvider.getProcessor(processorName);
        return processor.process(searchContext, queryModel);
    }
}
