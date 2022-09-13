package com.dmetasoul.metaspore.recommend.operator;

import com.dmetasoul.metaspore.serving.FeatureTable;
import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import org.apache.arrow.vector.complex.StructVector;
import org.apache.arrow.vector.complex.impl.NullableStructWriter;
import org.springframework.util.Assert;

@Slf4j
@Data
public class StructOperator extends ArrowOperator {
    @Override
    public boolean set(FeatureTable featureTable, int index, String col, Object value) {
        StructVector structVector = featureTable.getVector(col);
        Assert.notNull(structVector, "structVector is not null at col:" + col);
        if (value == null) {
            structVector.setNull(index);
        } else {
            NullableStructWriter writer = structVector.getWriter();
            writer.setPosition(index);
            log.info("struct vector: {}", structVector.getField().getChildren());
            writeStruct(writer, value, structVector.getField().getChildren(), structVector.getAllocator());
        }
        featureTable.setRowCount(index+1);
        return true;
    }
}
