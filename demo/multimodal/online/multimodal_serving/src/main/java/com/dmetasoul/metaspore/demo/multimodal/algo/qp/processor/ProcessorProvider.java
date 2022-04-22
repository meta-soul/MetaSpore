package com.dmetasoul.metaspore.demo.multimodal.algo.qp.processor;

import org.springframework.stereotype.Component;

import java.util.*;

@Component
public class ProcessorProvider {
    private final Map<String, Processor> processorMap;

    public ProcessorProvider(List<Processor> processors) {
        this.processorMap = new HashMap<>();
        processors.forEach(x -> processorMap.put(x.getClass().getSimpleName(), x));
    }

    public Processor getProcessor(String name) {
        return processorMap.get(name);
    }

    public Collection<Processor> getProcessors(Collection<String> names) {
        if (names == null) {
            return Collections.emptyList();
        }

        ArrayList<Processor> processors = new ArrayList<>();
        for (String name : names) {
            processors.add(getProcessor(name));
        }

        return processors;
    }
}
