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

import re

def is_valid_qualified_name(name):
    pattern = r'^[A-Za-z_.\-][A-Za-z0-9_.\-]*$'
    match = re.match(pattern, name)
    return match is not None

def simplify_name(name, ref_name):
    i = 0
    while i < len(name) and i < len(ref_name) and name[i] == ref_name[i]:
        i += 1
    j = len(name)
    k = len(ref_name)
    while i < j and i < k and name[j - 1] == ref_name[k - 1]:
        j -= 1
        k -= 1
    return name[i:j]

def get_words(name):
    word_regex = '[A-Za-z][a-z0-9]*'
    upper_case_word_regex = '[A-Z]+'
    upper_case_identifier_regex = '_*[A-Z]+(_+[A-Z]+)*_*$'
    if re.match(upper_case_identifier_regex, name):
        return re.findall(upper_case_word_regex, name)
    else:
        return re.findall(word_regex, name)

def to_lower_snake_case(name):
    return '_'.join(word.lower() for word in get_words(name))
