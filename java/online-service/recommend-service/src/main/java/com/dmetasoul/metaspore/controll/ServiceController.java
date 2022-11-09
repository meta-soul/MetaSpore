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

import com.dmetasoul.metaspore.baseservice.RecommendService;
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
import com.dmetasoul.metaspore.recommend.Layer;
import com.dmetasoul.metaspore.recommend.Scene;
import com.dmetasoul.metaspore.recommend.Service;
import com.dmetasoul.metaspore.recommend.Experiment;
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

import static org.springframework.web.bind.annotation.RequestMethod.POST;

/**
 * 用于实现restfull接口
 * Created by @author qinyy907 in 14:24 22/07/15.
 */
@Slf4j
@RestController
@RequestMapping("/service")
public class ServiceController {

    @Autowired
    private RecommendService recommendService;
    /**
     * 用于实现restfull接口 /service/get/{task}
     *
     * @param task 需要调用的任务名称，需要在配置中事先定义好，不可为空
     * @param req  需要传递给任务task的请求参数
     * @return ServiceResult 包含状态码，错误信息以及数据结果
     * Created by @author qinyy907 in 14:24 22/07/15.
     */
    @SuppressWarnings("unchecked")
    @RequestMapping(value = "/get/{task}", method = POST, produces = "application/json")
    public ServiceResult getDataServiceResult(@PathVariable String task, @RequestBody Map<String, Object> req) {
        return recommendService.getDataServiceResult(task, req);
    }

    @SneakyThrows
    @RequestMapping(value = "/recommend/{task}", method = POST, produces = "application/json")
    public ServiceResult getRecommendResult(@PathVariable String task, @RequestBody Map<String, Object> req) {
        return recommendService.getRecommendResult(task, req);
    }

    /**
     * 用于实现restfull接口 /service/recommend/{scene}/{id}
     * req 设置debug字段输出各个阶段的输出结果
     *
     * @param scene 需要推荐的场景名称，需要在配置中事先定义好，不可为空
     * @param id    需要被推荐用户的id，比如user id，不可为空
     * @param req   需要传递给推荐任务的请求参数，可为空
     * @return ServiceResult 包含状态码，错误信息以及数据结果
     * Created by @author qinyy907 in 14:24 22/07/15.
     */
    @RequestMapping(value = "/recommend/{scene}/{id}", method = POST, produces = "application/json")
    public ServiceResult recommend(@PathVariable String scene, @PathVariable String id, @RequestBody Map<String, Object> req) {
        return recommendService.recommend(scene, id, req);
    }

    // add cache later
    @RequestMapping(value = "/itemSummary/{item_key}/{id}", method = POST, produces = "application/json")
    public ServiceResult itemSummary(@PathVariable String item_key, @PathVariable String id, @RequestBody Map<String, Object> req) {
        return recommendService.itemSummary(item_key, id, req);
    }
}
