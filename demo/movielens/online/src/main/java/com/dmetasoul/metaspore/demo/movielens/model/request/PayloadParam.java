package com.dmetasoul.metaspore.demo.movielens.model.request;

import com.fasterxml.jackson.annotation.JsonProperty;
import com.google.common.collect.Maps;

import java.util.Map;

public class PayloadParam {
    @JsonProperty("useDebug")
    public Boolean useDebug = Boolean.FALSE;

    @JsonProperty("useDiversify")
    public Boolean useDiversify = Boolean.FALSE;

    @JsonProperty("specifiedLayerAndExperiment")
    public Map<String, String> specifiedLayerAndExperiment = Maps.newHashMap();

    @Override
    public String toString() {
        return "PayloadParameter{" +
                "useDebug=" + useDebug +
                ", useDiversify=" + useDiversify +
                ", specifiedLayerAndExperiment=" + specifiedLayerAndExperiment +
                '}';
    }
}

