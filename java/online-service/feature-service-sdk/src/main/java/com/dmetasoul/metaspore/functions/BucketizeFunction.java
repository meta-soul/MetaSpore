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
package com.dmetasoul.metaspore.functions;


import com.dmetasoul.metaspore.annotation.FeatureAnnotation;
import com.dmetasoul.metaspore.common.CommonUtils;
import com.dmetasoul.metaspore.configure.FieldAction;
import com.dmetasoul.metaspore.configure.FieldInfo;
import com.dmetasoul.metaspore.data.TableData;
import com.google.common.collect.Lists;
import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.collections4.CollectionUtils;
import org.apache.commons.lang3.StringUtils;
import org.apache.commons.lang3.Validate;
import org.apache.commons.lang3.math.NumberUtils;

import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutorService;

@Slf4j
@FeatureAnnotation("bucket")
public class BucketizeFunction implements Function {
    private final static String NAMEBINS = "bins";
    private final static String NAMEMIN = "min";
    private final static String NAMEMAX = "max";
    private final static String NAMERANGES = "ranges";

    private int bins = 10;
    private int min = 0;
    private int max = 120;

    private List<Number> ranges = Lists.newArrayList();

    @Override
    public boolean process(@NonNull TableData fieldTableData,
                           @NonNull FieldAction config, @NonNull ExecutorService taskPool) {
        Map<String, Object> options = config.getOptions();
        bins = CommonUtils.getField(options, NAMEBINS, bins);
        min = CommonUtils.getField(options, NAMEMIN, min);
        max = CommonUtils.getField(options, NAMEMAX, max);
        String rangeStr = CommonUtils.getField(options, NAMERANGES, "[]");
        ranges = parseRanges(rangeStr);
        if (CollectionUtils.isNotEmpty(config.getNames())) {
            int fieldSize = 0;
            if (CollectionUtils.isNotEmpty(config.getFields())) {
                fieldSize = config.getFields().size();
            }
            for (int i = 0; i < config.getNames().size(); ++i) {
                List<Object> result = Lists.newArrayList();
                FieldInfo fieldInfo = config.getInputFields().get(i);
                if (fieldInfo != null) {
                    for (int j = 0; j < fieldTableData.getData().size(); ++j) {
                        Object o = fieldTableData.getValue(j, fieldInfo);
                        Validate.isInstanceOf(Number.class, o, "value must be number!");
                        if (CollectionUtils.isNotEmpty(ranges)) {
                            result.add(bucket((Number) o, ranges));
                        } else {
                            result.add(bucket((Number) o, bins, max, min));
                        }
                    }
                }
                fieldTableData.addValueList(config.getNames().get(i), result);
            }
        }
        return true;
    }

    private List<Number> parseRanges(String rangeStr) {
        List<Number> res = Lists.newArrayList();
        if (StringUtils.isNotEmpty(rangeStr)) {
            int start = 0;
            int end = rangeStr.length();
            if (rangeStr.startsWith("[")) start += "[".length();
            if (rangeStr.endsWith("]")) end -= "]".length();
            String[] arr = rangeStr.substring(start, end).split(",");
            for (String s : arr) {
                res.add(NumberUtils.createNumber(s.strip()));
            }
        }
        return res;
    }

    private int bucket(Number value, Number bins, Number max, Number min) {
        Validate.isTrue(!bins.equals(0.0), "bins not zero");
        if (value.doubleValue() < min.doubleValue()) {
            return 0;
        }
        if (value.doubleValue() > max.doubleValue()) {
            return (int) ((max.doubleValue() - min.doubleValue()) / bins.doubleValue());
        }
        return (int) ((value.doubleValue() - min.doubleValue()) / bins.doubleValue());
    }

    private int bucket(Number value, List<Number> ranges) {
        Validate.isTrue(!ranges.isEmpty(), "ranges not empty");
        for (int i = 0; i < ranges.size(); ++i) {
            Number item = ranges.get(i);
            if (value.doubleValue() < item.doubleValue() && i == 0
                    || value.doubleValue() >= item.doubleValue() && i == ranges.size() - 1) {
                return i;
            }
            if (value.doubleValue() >= item.doubleValue() && value.doubleValue() < ranges.get(i + 1).doubleValue()) {
                return i;
            }
        }
        return 0;
    }
}
