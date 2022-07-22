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
package com.dmetasoul.metaspore.recommend.recommend;

import com.dmetasoul.metaspore.recommend.annotation.BucketizerAnnotation;
import com.dmetasoul.metaspore.recommend.common.SpringBeanUtil;
import com.dmetasoul.metaspore.recommend.configure.RecommendConfig;
import com.dmetasoul.metaspore.recommend.data.DataContext;
import com.dmetasoul.metaspore.recommend.data.DataResult;
import com.dmetasoul.metaspore.recommend.data.ServiceRequest;
import com.dmetasoul.metaspore.recommend.bucketizer.LayerBucketizer;
import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.collections4.MapUtils;

import java.util.List;
import java.util.Map;

@SuppressWarnings("rawtypes")
@Data
@Slf4j
public class Layer {

    private RecommendConfig.Layer layer;
    private LayerBucketizer bucketizer;

    public void init() {
        bucketizer = getLayerBucketizer(layer);
        if (bucketizer == null) {
            log.error("layer bucketizer：{} init fail！", layer.getBucketizer());
        }
    }

    public LayerBucketizer getLayerBucketizer(RecommendConfig.Layer layer) {
        LayerBucketizer layerBucketizer = (LayerBucketizer) SpringBeanUtil.getBean(layer.getBucketizer());
        if (layerBucketizer == null || !layerBucketizer.getClass().isAnnotationPresent(BucketizerAnnotation.class)) {
            log.error("the layer.getBucketizer:{} load fail!", layer.getBucketizer());
            return null;
        }
        layerBucketizer.init(layer);
        return layerBucketizer;
    }


    public DataResult process(ServiceRequest request, DataContext context) {
        DataResult result = null;
        String experiment = bucketizer.toBucket(context);
        result = new DataResult();
        return result;
    }
}
