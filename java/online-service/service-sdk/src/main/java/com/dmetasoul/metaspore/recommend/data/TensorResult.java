//
// Copyright 2022 DMetaSoul
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
package com.dmetasoul.metaspore.recommend.data;

import com.dmetasoul.metaspore.serving.ArrowTensor;
import lombok.Data;
import lombok.extern.slf4j.Slf4j;

import java.util.ArrayList;
import java.util.List;

/**
 * 用于保存Tensor 模型服务结果
 * Created by @author qinyy907 in 14:24 22/07/15.
 */
@Slf4j
@Data
public class TensorResult extends DataResult {
    private String fieldName;
    private int index = -1;
    private ArrowTensor tensor;

    public <T> List<List<T>> get() {
        if (tensor == null) {
            throw new IllegalArgumentException("tensor or shape is null");
        }
        ArrowTensor.TensorAccessor<T> accessor = getTensorAccessor();
        long[] shape = tensor.getShape();
        if (shape.length != 2) {
            throw new IllegalArgumentException("Shape length must equal to 2 (batch, vector dim). shape.length: " + shape.length);
        }
        List<List<T>> vectors = new ArrayList<>();
        for (int i = 0; i < shape[0]; i++) {
            List<T> vector = new ArrayList<>();
            for (int j = 0; j < shape[1]; j++) {
                vector.add(accessor.get(i, j));
            }
            vectors.add(vector);
        }
        return vectors;
    }

    public <T> List<T> getData(int targetIndex) {
        if (tensor == null) {
            throw new IllegalArgumentException("tensor or shape is null");
        }
        ArrowTensor.TensorAccessor<T> accessor = getTensorAccessor();
        long[] shape = tensor.getShape();
        if (targetIndex < 0 || targetIndex >= shape.length) {
            throw new IllegalArgumentException("Target index is out of shape scope. targetIndex: " + targetIndex);
        }
        List<T> scores = new ArrayList<>();
        for (int i = 0; i < shape[0]; i++) {
            scores.add(accessor.get(i, targetIndex));
        }
        return scores;
    }

    @SuppressWarnings("unchecked")
    private <T> ArrowTensor.TensorAccessor<T> getTensorAccessor() {
        if (tensor == null) throw new IllegalArgumentException("tensor is null");
        if (tensor.isFloatTensor()) {
            return (ArrowTensor.TensorAccessor<T>) tensor.getFloatData();
        } else if (tensor.isDoubleTensor()) {
            return (ArrowTensor.TensorAccessor<T>) tensor.getDoubleData();
        } else if (tensor.isLongTensor()) {
            return (ArrowTensor.TensorAccessor<T>) tensor.getLongData();
        } else {
            return (ArrowTensor.TensorAccessor<T>) tensor.getIntData();
        }
    }

    public boolean isNull() {
        return tensor == null;
    }
}
