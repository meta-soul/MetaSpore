import json
from typing import List, Dict, Optional

class Base(object):

    def __init__(self, **kwargs):
        pass

    def __str__(self):
        return str(self.to_dict())

    def to_dict(self):
        return {}

    def to_json(self):
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_json(cls, json_str):
        return cls(**json.loads(json_str))

    @classmethod
    def save_to_json(cls, instances, json_file):
        with open(json_file, 'w', encoding='utf8') as f:
            for i in instances:
                print(i.to_json(), file=f)

    @classmethod
    def load_from_json(cls, json_file):
        data = []
        with open(json_file, 'r', encoding='utf8') as fin:
            for line in fin:
                line = line.strip('\r\n')
                if not line:
                    continue
                x = json.loads(line)
                data.append(cls(**x))
        return data


class User(Base):

    def __init__(self, user_id: str, tags: Optional[List[str]]=[], tag_weights: Optional[List[float]]=[], **kwargs):
        self.user_id = user_id
        self.tags = tags
        self.tag_weights = tag_weights if tag_weights else [1.0 for i in range(len(tags))]

    def to_dict(self):
        return {
            'user_id': self.user_id,
            'tags': self.tags,
            'tag_weights': self.tag_weights
        }


class Item(Base):

    def __init__(self, item_id: str, 
            title: Optional[str]="", content: Optional[str]="", 
            category: Optional[str]="", tags: Optional[List[str]]=[], weight: Optional[float]=0.0, **kwargs):
        self.item_id = item_id
        self.title = title
        self.content = content
        self.category = category
        self.tags = tags
        self.weight = weight

    def to_dict(self):
        return {
            'item_id': self.item_id,
            'title': self.title,
            'content': self.content,
            'category': self.category,
            'tags': self.tags,
            'weight': self.weight
        }


class Action(Base):

    def __init__(self, scene_id: str, user_id: str, item_id: str, action_type: str, action_time: int, action_value: float,
            extra: Optional[Dict]={}):
        self.scene_id = scene_id
        self.user_id = user_id
        self.item_id = item_id
        self.action_type = action_type
        self.action_time = int(action_time)
        self.action_value = float(action_value)
        self.extra = extra
        self.user = None
        self.item = None

    def to_dict(self):
        return {
            'scene_id': self.scene_id,
            'user_id': self.user_id,
            'item_id': self.item_id,
            'action_type': self.action_type,
            'action_time': self.action_time,
            'action_value': self.action_value,
            'extra': self.extra
        }

    @staticmethod
    def join_user(actions, users):
        user_map = {u.user_id:u for u in users}
        for action in actions:
            action.user = user_map.get(action.user_id, None)
        return actions

    @staticmethod
    def join_item(actions, items):
        item_map = {i.item_id:i for i in items}
        for action in actions:
            action.item = item_map.get(action.item_id, None)
        return actions

    @staticmethod
    def join(actions, users=None, items=None):
        if users is not None:
            actions = Action.join_user(actions, users)
        if items is not None:
            actions = Action.join_item(actions, items)
        return actions

    @staticmethod
    def group_by_user_id(actions, scene_id, action_type, action_value_min=0.0, action_value_max=float('inf'), sortby='action_time', reverse=False):
        assert sortby in ['action_time', 'action_value']
        data = {}
        for action in actions:
            if action.scene_id != scene_id:
                continue
            if action.action_type != action_type:
                continue
            if action.action_value < action_value_min:
                continue
            if action.action_value > action_value_max:
                continue
            user_id = action.user_id
            if user_id not in data:
                data[user_id] = []
            data[user_id].append(action)
        for user_id, user_actions in data.items():
            user_actions = sorted(user_actions, key=lambda x:getattr(x, sortby), reverse=reverse)
            yield user_id, user_actions
