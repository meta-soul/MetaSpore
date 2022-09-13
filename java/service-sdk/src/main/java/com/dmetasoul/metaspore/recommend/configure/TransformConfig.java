package com.dmetasoul.metaspore.recommend.configure;

import lombok.Data;

import java.util.Map;

@Data
public class TransformConfig {
    private String name;
    private Map<String, Object> option;
}
