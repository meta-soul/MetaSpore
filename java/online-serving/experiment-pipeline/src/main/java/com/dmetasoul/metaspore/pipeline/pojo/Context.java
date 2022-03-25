package com.dmetasoul.metaspore.pipeline.pojo;

import lombok.Data;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

@Data
public class Context {

    private List<LayerContext> layerContexts = new ArrayList<>();

    public Map<String, Object> sceneArgs = new HashMap<>();

    public void add(LayerContext layerContext) {
        layerContexts.add(layerContext);
    }

}
