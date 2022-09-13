package com.dmetasoul.metaspore.recommend.operator;

import com.dmetasoul.metaspore.serving.FeatureTable;
import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import org.apache.arrow.vector.complex.MapVector;
import org.apache.arrow.vector.complex.impl.UnionListWriter;
import org.springframework.util.Assert;

import java.util.Map;

@Slf4j
@Data
public class MapOperator<K, V> extends ArrowOperator {
    @Override
    public boolean set(FeatureTable featureTable, int index, String col, Object value) {
        if (value != null && !(value instanceof Map)) {
            log.error("set featureTable fail! value type is not match!");
            return false;
        }
        MapVector mapVector = featureTable.getVector(col);
        Assert.notNull(mapVector, "mapvector is not null at col:" + col);
        if (value == null) {
            mapVector.setNull(index);
        } else {
            @SuppressWarnings("unchecked") Map<K, V> data = (Map<K, V>) value;
            UnionListWriter writer = mapVector.getWriter();
            writer.setPosition(index);
            writeMap(writer, data, mapVector.getField().getChildren(), mapVector.getAllocator());
        }
        featureTable.setRowCount(index+1);
        return true;
    }
}
