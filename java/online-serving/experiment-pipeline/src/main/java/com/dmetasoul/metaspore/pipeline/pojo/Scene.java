package com.dmetasoul.metaspore.pipeline.pojo;

import lombok.Data;
import org.springframework.stereotype.Component;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

@Data
@Component
public class Scene {
    private String name;
    private Map<String, Object> sceneArgs = new HashMap<>();
    private List<Layer> layers;
}


