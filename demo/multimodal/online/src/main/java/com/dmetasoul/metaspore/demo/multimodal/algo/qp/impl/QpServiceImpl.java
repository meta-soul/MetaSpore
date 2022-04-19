package com.dmetasoul.metaspore.demo.multimodal.algo.qp.impl;

import com.dmetasoul.metaspore.demo.multimodal.algo.qp.QpService;
import com.dmetasoul.metaspore.demo.multimodal.algo.qp.processor.Processor;
import com.dmetasoul.metaspore.demo.multimodal.algo.qp.processor.ProcessorProvider;
import com.dmetasoul.metaspore.demo.multimodal.model.QueryModel;
import com.dmetasoul.metaspore.demo.multimodal.model.SearchContext;
import org.springframework.stereotype.Service;

import java.io.IOException;

@Service
public class QpServiceImpl implements QpService {

    private final ProcessorProvider processorProvider;

    public QpServiceImpl(ProcessorProvider processorProvider) {
        this.processorProvider = processorProvider;
    }

    @Override
    public void process(SearchContext searchContext, QueryModel queryModel) throws IOException {
        String processorName = searchContext.getQpQueryProcessorModelName();
        Processor processor = processorProvider.getProcessor(processorName);
        processor.process(searchContext, queryModel);
    }
}
