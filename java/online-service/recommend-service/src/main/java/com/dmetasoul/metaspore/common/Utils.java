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

package com.dmetasoul.metaspore.common;


import com.google.common.collect.Maps;
import lombok.SneakyThrows;
import lombok.extern.slf4j.Slf4j;
import org.springframework.util.StopWatch;

import java.io.BufferedReader;
import java.io.File;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.Map;

/**
 * 工具类
 * Created by @author qinyy907 in 14:24 22/07/15.
 */

@Slf4j
public class Utils {

    public static String getCacheKey(String name, String key) {
        return String.format("CACHE_%s_%s", name, key);
    }


    public static Map<String, Long> getTimeRecords(StopWatch stopWatch) {
        Map<String, Long> records = Maps.newHashMap();
        stopWatch.getTaskInfo();
        for (StopWatch.TaskInfo info : stopWatch.getTaskInfo()) {
            records.put(info.getTaskName(), info.getTimeMillis());
        }
        return records;
    }

    public static double getFinalRetrievalScore(Double originalScore, Double maxScore, int algoLevel) {
        double EPSILON = 0.001;
        return originalScore / (maxScore + EPSILON) + algoLevel;
    }

    @SneakyThrows
    public static String runCmd(String cmd, File dir) {
        StringBuilder result = new StringBuilder();
        Process process = Runtime.getRuntime().exec(cmd, null, dir);
        try (BufferedReader bufrIn = new BufferedReader(new InputStreamReader(process.getInputStream(), StandardCharsets.UTF_8));
             BufferedReader bufrError = new BufferedReader(new InputStreamReader(process.getErrorStream(), StandardCharsets.UTF_8))) {
            process.waitFor();
            String line;
            while ((line = bufrIn.readLine()) != null) {
                result.append(line).append('\n');
            }
            while ((line = bufrError.readLine()) != null) {
                result.append(line).append('\n');
            }
        } finally {
            if (process != null) {
                process.destroy();
            }
        }
        return result.toString();
    }
}
