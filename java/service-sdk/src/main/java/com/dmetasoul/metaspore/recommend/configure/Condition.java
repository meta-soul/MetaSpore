package com.dmetasoul.metaspore.recommend.configure;

import com.dmetasoul.metaspore.recommend.enums.JoinTypeEnum;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.collections4.MapUtils;

import java.util.Map;

@Data
@AllArgsConstructor
@Slf4j
public class Condition {
    FieldInfo left;
    FieldInfo right;
    JoinTypeEnum type;
    private Condition() {
        type = JoinTypeEnum.INNER;
    }

    public static Condition reverse(Condition cond) {
        Condition condition = new Condition();
        condition.left = cond.getRight();
        condition.right = cond.getLeft();
        if (cond.getType() == JoinTypeEnum.LEFT) {
            condition.setType(JoinTypeEnum.RIGHT);
        }
        if (cond.getType() == JoinTypeEnum.RIGHT) {
            condition.setType(JoinTypeEnum.LEFT);
        }
        return condition;
    }

    public static Condition create(Map<String, String> data) {
        if (MapUtils.isEmpty(data)) {
            log.error("feature condition config is wrong");
            return null;
        }
        if ((data.containsKey("type") && data.size() == 2) || data.size() == 1) {
            Condition condition = new Condition();
            data.forEach((key, value) -> {
                if (key.equals("type")) {
                    condition.type = JoinTypeEnum.getEnumByName(value);
                } else {
                    condition.left = FieldInfo.create(key);
                    condition.right = FieldInfo.create(value);
                }
            });
            if (condition.isInvalid()) {
                return null;
            }
            return condition;
        }
        return null;
    }

    public boolean isInvalid() {
        return left == null || right == null;
    }
}