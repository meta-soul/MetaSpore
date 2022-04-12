package com.dmetasoul.metaspore.demo.multimodel.model.request;

import com.fasterxml.jackson.annotation.JsonProperty;
import com.google.common.collect.Maps;

import java.util.Map;

public class PayloadParam {
    @JsonProperty("query")
    public String query;

    @JsonProperty("useDebug")
    public Boolean useDebug = Boolean.FALSE;

    @JsonProperty("useDiversify")
    public Boolean useDiversify = Boolean.FALSE;

    @JsonProperty("specifiedLayerAndExperiment")
    public Map<String, String> specifiedLayerAndExperiment = Maps.newHashMap();

    @Override
    public String toString() {
        return "PayloadParam{" +
                "query='" + query + '\'' +
                ", useDebug=" + useDebug +
                ", useDiversify=" + useDiversify +
                ", specifiedLayerAndExperiment=" + specifiedLayerAndExperiment +
                '}';
    }
}
