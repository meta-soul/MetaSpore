#
# Copyright 2022 DMetaSoul
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

import uuid
import requests
from flask import render_template
from flask import Flask, jsonify, request

from service import txt2img_search_service, txt2txt_search_service, img2img_search_service, img2txt_search_service

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = '/data/xwb/datasets/uploads/imgs'

def get_extname(filename):
    if '.' not in filename:
        return None
    ext = filename.rsplit('.', 1)[1].lower()
    if ext not in {'png', 'jpg', 'jpeg', 'gif'}:
        return None
    return ext


@app.route('/')
def index():
    pages = ['txt2txt', 'txt2img']
    #pages = ['txt2txt']
    return render_template('index.html', pages=pages)

@app.route('/txt', methods=['GET', 'POST'])
def txt_search_index():
    results = []
    query = ''
    search_type = request.args.get('search_type', 'txt')
    if request.method == 'POST':
        search_type = request.form['search_type']
        query = request.form['query']
        if search_type == 'img':
            results = txt2img_search_service(query)
        else:
            results = txt2txt_search_service(query)
    return render_template('txt_search_index.html', results=results, query=query, search_type=search_type)

@app.route('/img', methods=['GET', 'POST'])
def img_search_index():
    results = []
    search_type = request.args.get('search_type', 'txt')
    if request.method == 'POST':
        search_type = request.form['search_type']
        #filename = request.files['image'].filename
        #extname = get_extname(filename)
        if search_type == 'img':
            results = img2img_search_service(request.files['image'].stream)
        else:
            results = img2txt_search_service(request.files['image'].stream)
    return render_template('img_search_index.html', results=results, search_type=search_type)
