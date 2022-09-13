package com.dmetasoul.metaspore.recommend.data;

import lombok.Data;

@Data
public class IndexData {
    public static final int EntireColumn = -1;
    private int index;
    private Object val;

    public IndexData(int index, Object val) {
        this.index = index;
        this.val = val;
    }

    @SuppressWarnings("unchecked")
    public <T> T getVal() {
        return (T) val;
    }

    public boolean isAggregate() {return index == EntireColumn;}
}
