package com.dmetasoul.metaspore.recommend.controll;

import com.dmetasoul.metaspore.recommend.configure.TaskFlowConfig;
import com.dmetasoul.metaspore.recommend.data.DataContext;
import com.dmetasoul.metaspore.recommend.data.DataResult;
import com.dmetasoul.metaspore.recommend.data.ServiceResult;
import com.dmetasoul.metaspore.recommend.TaskFlow;
import com.dmetasoul.metaspore.recommend.dataservice.DataService;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.StringUtils;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.Map;

import static org.springframework.web.bind.annotation.RequestMethod.POST;

@Slf4j
@RestController
@RequestMapping("/service")
public class ServiceController {

    @Autowired
    public TaskFlowConfig taskFlowConfig;

    @Autowired
    public TaskFlow taskFlow;

    @RequestMapping(value = "/get/{task}", method = POST, produces = "application/json")
    public ServiceResult getTaskData(@PathVariable String task, @RequestBody Map<String, Object> req) {
        log.info("get task :{}, request:{}", task, req);
        DataService taskService = taskFlow.getTaskService(task);
        if (taskService == null) {
            return ServiceResult.of(-1, "taskService is not exist!");
        }
        DataResult result = taskService.execute(new DataContext(req));
        if (result == null) {
            return ServiceResult.of(-1, "taskService execute fail!");
        }
        if (!result.isVaild()) {
            return ServiceResult.of(-1, "taskService result is invalid!");
        }
        return ServiceResult.of(result);
    }

    @RequestMapping(value = "/recommend/{scene}/{id}", method = POST, produces = "application/json")
    public ServiceResult recommend(@PathVariable String scene, @PathVariable String id, @RequestBody Map<String, Object> req) {
        DataService taskService = taskFlow.getTaskService(scene);
        if (taskService == null) {
            return ServiceResult.of(-1, String.format("scene:%s is not support!", scene));
        }
        DataContext context = new DataContext(req);
        if (StringUtils.isEmpty(id)) {
            return ServiceResult.of(-1, String.format("scene:%s recommend need id, eg:userId!", scene));
        }
        context.setId(id);
        DataResult result = taskService.execute(context);
        if (result == null) {
            return ServiceResult.of(-1, "scene execute fail!");
        }
        if (!result.isVaild()) {
            return ServiceResult.of(-1, "scene result is invalid!");
        }
        return ServiceResult.of(result, id);
    }
}
