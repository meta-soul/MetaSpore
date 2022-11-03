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
package com.dmetasoul.metaspore.controll;

import com.dmetasoul.metaspore.baseservice.TaskServiceRegister;
import com.dmetasoul.metaspore.common.CommonUtils;
import com.dmetasoul.metaspore.common.Utils;
import com.dmetasoul.metaspore.configure.AlgoTransform;
import com.dmetasoul.metaspore.configure.RecommendConfig;
import com.dmetasoul.metaspore.configure.TaskFlowConfig;
import com.dmetasoul.metaspore.data.DataContext;
import com.dmetasoul.metaspore.data.DataResult;
import com.dmetasoul.metaspore.data.ServiceRequest;
import com.dmetasoul.metaspore.data.ServiceResult;
import com.dmetasoul.metaspore.dataservice.DataService;
import com.dmetasoul.metaspore.enums.DataTypeEnum;
import com.dmetasoul.metaspore.recommend.Experiment;
import com.dmetasoul.metaspore.recommend.Layer;
import com.dmetasoul.metaspore.recommend.Scene;
import com.dmetasoul.metaspore.recommend.Service;
import com.google.common.collect.Lists;
import lombok.SneakyThrows;
import lombok.extern.slf4j.Slf4j;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.commons.collections4.CollectionUtils;
import org.apache.commons.lang3.StringUtils;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.util.StopWatch;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.List;
import java.util.Map;
import java.util.UUID;

import static org.springframework.web.bind.annotation.RequestMethod.GET;
import static org.springframework.web.bind.annotation.RequestMethod.POST;

/**
 * 用于实现restfull接口
 * Created by @author qinyy907 in 14:24 22/07/15.
 */
@Slf4j
@RestController
@RequestMapping
public class SageMakerController {

    @Autowired
    public TaskFlowConfig taskFlowConfig;

    @Autowired
    public TaskServiceRegister taskServiceRegister;

    @RequestMapping(value = "/ping", method = GET, produces = "application/json")
    public ServiceResult ping() {
        return ServiceResult.of(0, "recommend service is ok!");
    }

    @RequestMapping(value = "/invocations", method = POST, produces = "application/json")
    public ServiceResult invocations(@RequestBody Map<String, Object> req) {
        return ServiceResult.of(0, "recommend service invocations todo!");
    }
}
