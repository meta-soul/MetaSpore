package com.dmetasoul.metaspore.example.layer;

import lombok.Data;

@Data
public class FirstLayerPojo {
    private String userId;
    private String milvus;
    private String milvus2;
    private String milvus3;
    private String milvus4;

    public FirstLayerPojo(String userId) {
        this.userId = userId;
    }
}
