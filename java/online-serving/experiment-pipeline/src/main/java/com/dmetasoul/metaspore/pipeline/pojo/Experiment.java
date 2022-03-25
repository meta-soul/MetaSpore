package com.dmetasoul.metaspore.pipeline.pojo;

import lombok.Data;

import java.util.HashMap;
import java.util.Map;

@Data
public class Experiment {
    private String layerName;
    private String experimentName;
    private Map<String, Object> experimentArgs = new HashMap<>();
}


