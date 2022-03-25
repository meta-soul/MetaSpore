package com.dmetasoul.metaspore.pipeline;

import java.util.Map;

public interface Scene {
    Object run(Object in);

    Object runDebug(Object in, Map<String, String> specifiedLayerAndExperiment);

}
