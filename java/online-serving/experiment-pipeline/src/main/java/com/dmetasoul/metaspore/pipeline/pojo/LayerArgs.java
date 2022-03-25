package com.dmetasoul.metaspore.pipeline.pojo;

import lombok.Data;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

@Data
public class LayerArgs {
    private List<NormalLayerArgs> normalLayerArgs = new ArrayList<>();
    private Map<String, Object> extraLayerArgs = new HashMap<>();

    public LayerArgs(List<NormalLayerArgs> normalLayerArgs, Map<String, Object> extraLayerArgs) {
        this.normalLayerArgs = normalLayerArgs;
        this.extraLayerArgs = extraLayerArgs;
    }
}
