package com.dmetasoul.metaspore.pipeline.pojo;


import lombok.Data;

@Data
public class LayerContext<I, O> {
    private int layerNum;
    private String layerName;
    private String SplitExperimentName;
    private I input;
    private O output;
}