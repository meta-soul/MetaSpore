package com.dmetasoul.metaspore.recommend.data;

import com.dmetasoul.metaspore.recommend.enums.DataTypeEnum;
import com.dmetasoul.metaspore.recommend.operator.ArrowConv;
import lombok.Data;

@Data
public class IndexData {
    public static final int EntireColumn = -1;
    private int index;

    private DataTypeEnum type;
    private Object val;

    public IndexData(int index, Object val) {
        this.index = index;
        this.val = val;
    }

    @SuppressWarnings("unchecked")
    public <T> T getVal() {
        return (T) ArrowConv.convValue(type, val);
    }

    public boolean isAggregate() {return index == EntireColumn;}
}
