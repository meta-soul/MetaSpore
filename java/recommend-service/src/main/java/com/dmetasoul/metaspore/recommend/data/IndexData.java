package com.dmetasoul.metaspore.recommend.data;

import lombok.AllArgsConstructor;
import lombok.Data;

@Data
@AllArgsConstructor
public class IndexData {
    public static final int EntireColumn = -1;
    private int index;
    private Object val;

    public <T> T getVal() {
        return (T) val;
    }

    public boolean isAggregate() {return index == EntireColumn;}
}
