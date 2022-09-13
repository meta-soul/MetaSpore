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
package com.dmetasoul.metaspore.recommend.functions;


import com.dmetasoul.metaspore.recommend.annotation.FunctionAnnotation;
import com.dmetasoul.metaspore.recommend.common.CommonUtils;
import com.dmetasoul.metaspore.recommend.configure.FieldAction;
import com.dmetasoul.metaspore.recommend.data.FieldData;
import com.dmetasoul.metaspore.recommend.data.IndexData;
import com.dmetasoul.metaspore.recommend.functions.Function;
import com.google.common.collect.Lists;
import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.collections4.CollectionUtils;
import org.apache.commons.lang3.StringUtils;
import org.springframework.util.Assert;
import org.springframework.util.NumberUtils;

import javax.validation.constraints.NotEmpty;
import java.util.List;
import java.util.Map;

@Slf4j
@FunctionAnnotation("bucketize")
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
    public boolean process(@NotEmpty List<FieldData> fields, @NotEmpty List<FieldData> result, @NonNull FieldAction config) {
        Map<String, Object> options = config.getOptions();
        Assert.isTrue(CollectionUtils.isNotEmpty(fields) && fields.size() == 1, "input values size must eq 1");
        bins = CommonUtils.getField(options, NAMEBINS, bins);
        min = CommonUtils.getField(options, NAMEMIN, min);
        max = CommonUtils.getField(options, NAMEMAX, max);
        String rangeStr = CommonUtils.getField(options, NAMERANGES, "[]");
        ranges = parseRanges(rangeStr);
        FieldData fieldData = fields.get(0);
        List<IndexData> input = fieldData.getIndexValue();
        List<IndexData> res = Lists.newArrayList();
        for (IndexData o : input) {
            Assert.isInstanceOf(Number.class, o, "value must be number!");
            if (CollectionUtils.isNotEmpty(ranges)) {
                res.add(FieldData.create(o.getIndex(), bucket(o.getVal(), ranges)));
            } else {
                res.add(FieldData.create(o.getIndex(), bucket(o.getVal(), bins, max, min)));
            }
        }

        result.get(0).setIndexValue(res);
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
                res.add(NumberUtils.parseNumber(s.strip(), Number.class));
            }
        }
        return res;
    }

    private int bucket(Number value, Number bins, Number max, Number min) {
        Assert.isTrue(!bins.equals(0.0), "bins not zero");
        if (value.doubleValue() < min.doubleValue()) {
            return 0;
        }
        if (value.doubleValue() > max.doubleValue()) {
            return (int) ((max.doubleValue() - min.doubleValue()) / bins.doubleValue());
        }
        return (int) ((value.doubleValue() - min.doubleValue()) / bins.doubleValue());
    }

    private int bucket(Number value, List<Number> ranges) {
        Assert.isTrue(!ranges.isEmpty(), "ranges not empty");
        for (int i = 0; i < ranges.size(); ++i) {
            Number item = ranges.get(i);
            if (value.doubleValue() < item.doubleValue() && i == 0
                || value.doubleValue() >= item.doubleValue() && i == ranges.size() - 1) {
                return i;
            }
            if (value.doubleValue() >= item.doubleValue() && value.doubleValue() < ranges.get(i+1).doubleValue()) {
                return i;
            }
        }
        return 0;
    }
}
