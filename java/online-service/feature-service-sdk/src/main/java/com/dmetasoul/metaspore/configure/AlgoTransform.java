package com.dmetasoul.metaspore.configure;

import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.google.common.collect.Queues;
import com.google.common.collect.Sets;
import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.collections4.CollectionUtils;
import org.apache.commons.lang3.StringUtils;
import org.apache.commons.lang3.Validate;

import java.util.List;
import java.util.Map;
import java.util.Queue;
import java.util.Set;

@Slf4j
@Data
public class AlgoTransform extends TableInfo {
    private String name;
    private String taskName;
    private List<String> feature;
    private List<String> algoTransform;
    private List<FieldAction> fieldActions;
    private List<String> output;
    private Map<String, Object> options;

    private List<FieldAction> actionList;

    protected Map<String, String> columnRel;

    public void setFeature(List<String> list) {
        feature = list;
    }

    public void setFeature(String str) {
        feature = List.of(str);
    }

    public void setAlgoTransform(List<String> list) {
        algoTransform = list;
    }

    public void setAlgoTransform(String str) {
        algoTransform = List.of(str);
    }

    public boolean checkAndDefault() {
        if (!check()) {
            return false;
        }
        columnNames = output;
        columnMap = Maps.newHashMap();
        fieldMap = Maps.newHashMap();
        columnRel = Maps.newHashMap();
        actionList = Lists.newArrayList();
        Map<String, FieldAction> fieldActionMap = Maps.newHashMap();
        Queue<FieldAction> queue = Queues.newArrayDeque();
        Set<String> actionSet = Sets.newHashSet();
        Set<String> doneSet = Sets.newHashSet();
        if (CollectionUtils.isNotEmpty(fieldActions)) {
            for (FieldAction action : fieldActions) {
                Validate.isTrue(action.checkAndDefault(), "field action check!");
                for (int i = 0; i < action.getNames().size(); ++i) {
                    String name = action.getNames().get(i);
                    Validate.isTrue(!fieldActionMap.containsKey(name), "name must not be duplicate!");
                    fieldActionMap.put(name, action);
                    columnMap.put(name, getType(action.getTypes().get(i)));
                    fieldMap.put(name, getField(name, action.getTypes().get(i)));
                    columnRel.put(name, action.getNames().get(0));
                }
            }
        }
        for (String col : output) {
            if (actionSet.contains(col)) continue;
            FieldAction action = fieldActionMap.get(col);
            Validate.notNull(action, "output col must has Action colName:" + col);
            queue.offer(action);
            actionSet.addAll(action.getNames());
        }
        boolean flag;
        while (!queue.isEmpty()) {
            FieldAction action = queue.poll();
            flag = true;
            if (CollectionUtils.isNotEmpty(action.getInput())) {
                for (String item : action.getInput()) {
                    if (doneSet.contains(item)) {
                        continue;
                    }
                    flag = false;
                    if (actionSet.contains(item)) {
                        continue;
                    }
                    FieldAction fieldAction = fieldActionMap.get(item);
                    Validate.notNull(fieldAction, "FieldAction.input item must has Action item:" + item);
                    queue.offer(fieldAction);
                    actionSet.addAll(fieldAction.getNames());
                }
            }
            if (flag) {
                actionList.add(action);
                doneSet.addAll(action.getNames());
            } else {
                queue.offer(action);
            }
        }
        if (StringUtils.isEmpty(this.taskName)) {
            this.taskName = "AlgoTransform";
        }
        return true;
    }

    protected boolean check() {
        if (StringUtils.isEmpty(name) || CollectionUtils.isEmpty(output)) {
            log.error("AlgoInference config name, fieldActions must not be empty!");
            throw new IllegalStateException("AlgoInference config name, fieldActions must not be empty!");
        }
        return true;
    }
}