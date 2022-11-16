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
import com.dmetasoul.metaspore.data.ServiceResult;
import com.dmetasoul.metaspore.recommend.Scene;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.collections4.MapUtils;
import org.apache.commons.lang3.StringUtils;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.List;
import java.util.Map;
import java.util.Set;

import static com.dmetasoul.metaspore.baseservice.RecommendService.SPRING_CONFIG_NAME;
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
    private RecommendService recommendService;

    @Autowired
    public TaskServiceRegister taskServiceRegister;

    @RequestMapping(value = "/ping", method = GET, produces = "application/json")
    public ServiceResult ping() {
        return ServiceResult.of(0, "recommend service is ok!");
    }

    @RequestMapping(value = "/invocations", method = POST, produces = "application/json")
    public ServiceResult invocations(@RequestBody Map<String, Object> req) {
        String operator = CommonUtils.getField(req, "operator", "recommend");
        if (operator.equalsIgnoreCase("updateconfig")) {
            String config = CommonUtils.getField(req, "config");
            if (config == null) {
                return ServiceResult.of(-1, "no config arg in updateconfig!");
            }
            String configName = CommonUtils.getField(req, "configName", SPRING_CONFIG_NAME);
            String configFormat = CommonUtils.getField(req, "configFormat", "yaml");
            if (StringUtils.isEmpty(configName) || StringUtils.isEmpty(configFormat)) {
                return ServiceResult.of(-1, String.format("config name:%s or format:%s must not empty in updateconfig!", configName, configFormat));
            }
            Set<String> keys = recommendService.updateConfig(configName, config, configFormat);
            if (keys == null) {
                return ServiceResult.of(-1, String.format("updateconfig fail format: %s!", configFormat));
            }
            return ServiceResult.of(0, "update config successfully!")
                    .setInfo("config", config)
                    .setInfo("updateKeys", keys);
        } else if (operator.equalsIgnoreCase("loadmodel")) {
            Map<String, Boolean> res = recommendService.notifyToLoadModel(List.of(req));
            if (res.isEmpty()) {
                return ServiceResult.of(-1, "serving get fail!").setInfo(req);
            }
            for (Map.Entry<String, Boolean> entry : res.entrySet()) {
                if (!entry.getValue()) {
                    return ServiceResult.of(-1, "serving get fail!").setInfo(req);
                }
            }
            return ServiceResult.of(0, "serving load model resp successfully!").setInfo(req);
        } else if (operator.equalsIgnoreCase("feature")) {
            Map<String, Object> request = CommonUtils.getField(req, "request");
            if (MapUtils.isEmpty(request)) {
                return ServiceResult.of(-1, "no request arg in feature!");
            }
            String task = CommonUtils.getField(request, "task");
            if (StringUtils.isEmpty(task)) {
                return ServiceResult.of(-1, String.format("get feature task result need teskName: %s!", task));
            }
            try {
                return recommendService.getDataServiceResult(task, request);
            } catch (Exception ex) {
                log.error("feature exception: ", ex);
                return ServiceResult.of(-2, "service exec fail");
            }
        } else if (operator.equalsIgnoreCase("recommend")) {
            Map<String, Object> request = CommonUtils.getField(req, "request");
            if (MapUtils.isEmpty(request)) {
                return ServiceResult.of(-1, "no request arg in recommend!");
            }
            String scene = CommonUtils.getField(request, "scene");
            if (StringUtils.isEmpty(scene)) {
                return ServiceResult.of(-1, String.format("get recommend result need scene: %s!", scene));
            }
            Scene sceneService = taskServiceRegister.getScene(scene);
            if (sceneService == null) {
                return ServiceResult.of(-1, String.format("scene:%s is not support!", scene));
            }
            try {
                return recommendService.getRecommendResult(scene, request);
            } catch (Exception ex) {
                log.error("recommend exception: ", ex);
                return ServiceResult.of(-2, "service exec fail");
            }
        } else if (operator.equalsIgnoreCase("itemSummary")) {
            Map<String, Object> request = CommonUtils.getField(req, "request");
            if (MapUtils.isEmpty(request)) {
                return ServiceResult.of(-1, "no request arg in recommend!");
            }
            String item_key = CommonUtils.getField(request, "item_key", "item_id");
            String id = CommonUtils.getField(request, item_key);
            if (StringUtils.isEmpty(id)) {
                return ServiceResult.of(-1, String.format("itemSummary need %s: %s!", item_key, id));
            }
            try {
                return recommendService.itemSummary(item_key, id, request);
            } catch (Exception ex) {
                log.error("itemsummary exception: ", ex);
                return ServiceResult.of(-2, "service exec fail");
            }
        }
        return ServiceResult.of(-1, "no method support to " + operator);
    }
}
