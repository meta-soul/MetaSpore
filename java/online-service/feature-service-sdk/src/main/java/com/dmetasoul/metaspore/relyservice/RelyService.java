package com.dmetasoul.metaspore.relyservice;

import java.util.Map;

public interface RelyService extends AutoCloseable {
    void init(Map<String, Object> option);

    @Override
    void close();
}
