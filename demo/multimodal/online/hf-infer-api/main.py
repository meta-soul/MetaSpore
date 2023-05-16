#
# Copyright 2023 DMetaSoul
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from enum import Enum
from typing import Union

from fastapi import FastAPI
from pydantic import BaseModel

from models import TextCompletionModel, Text2ImageModel, Text2TextModel, TranslationModel

class TaskName(str, Enum):
    text_completion = "text-completion"
    text2text = "text-to-text"
    text2image = "text-to-image"
    text_translation = "text-translation"

class InferPayload(BaseModel):
    inputs: str
    args: str = "{}"


app = FastAPI()

@app.get("/")
async def home():
    return {"message": "Hello World"}

@app.post("/api/infer/{task_name}")
async def infer_api(task_name: TaskName, payload: InferPayload, model_type: Union[str, None]=None, model_name: Union[str, None]=None):

    if task_name is TaskName.text2image:
        return Text2ImageModel(model_name=model_name, model_type=model_type)(payload)
    elif task_name is TaskName.text_completion:
        return TextCompletionModel(model_name=model_name, model_type=model_type)(payload)
    elif task_name is TaskName.text2text:
        return Text2TextModel(model_name=model_name, model_type=model_type)(payload)
    elif task_name is TaskName.text_translation:
        return TranslationModel(model_name=model_name, model_type=model_type)(payload)
    return {"data": None}
