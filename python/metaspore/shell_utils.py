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

import sys
import string

BASH_SAFE_CHARS = frozenset(string.ascii_letters + string.digits + '+-_=/%@:,.')

def check_bash_string(item):
    if isinstance(item, str):
        return item
    elif isinstance(item, bytes):
        return item.decode()
    elif isinstance(item, (int, float)):
        return str(item)
    else:
        message = "'item' must be string or number; "
        message += "%r is not supported" % type(item)
        raise TypeError(message)

def escape_bash_string(item):
    item = check_bash_string(item)
    if not item:
        return "''"
    if all(c in BASH_SAFE_CHARS for c in item):
        return item
    if len(item) == 1:
        c = item[0]
        return '"\'"' if c == "'" else "'%c'" % c
    prev_index = None
    string = ''
    index = item.find('=')
    if index != -1 and all(c in BASH_SAFE_CHARS for c in item[:index]):
        index += 1
        string = item[:index]
        item = item[index:]
    for index, c in enumerate(item):
        if c == "'":
            if prev_index is not None:
                prev_index = None
                item += "'"
            string += '"\'"'
        else:
            if prev_index is None:
                prev_index = index
                string += "'"
            string += c
    if prev_index is not None:
        string += "'"
    return string

def escape_bash_command(command):
    if not isinstance(command, (list, tuple)):
        message = "'command' must be list or tuple; "
        message += "%r is not supported" % type(command)
        raise TypeError(message)
    if not command:
        message = "'command' can not be empty"
        raise ValueError(message)
    if len(command) == 1:
        return escape_bash_string(command[0])
    return ' '.join(escape_bash_string(x) for x in command)

def bash_escape(args):
    if not isinstance(args, (list, tuple)):
        return escape_bash_string(args)
    elif all(not isinstance(x, (list, tuple)) for x in args):
        return escape_bash_command(args)
    else:
        return '; '.join(escape_bash_command(x) for x in args)

def wrap_message(color, message, *, check_stderr=False):
    stream = sys.stderr if check_stderr else sys.stdout
    is_atty = getattr(stream, 'isatty', None)
    if is_atty and is_atty():
        message = '\033[%sm%s\033[m' % (color, message)
    return message

def log_message(color, message):
    message = wrap_message(color, message, check_stderr=True)
    print(message, file=sys.stderr)

def log_error(message):
    log_message('38;5;196', message)

def log_warning(message):
    log_message('38;5;051', message)

def log_info(message):
    log_message('38;5;231', message)

def log_debug(message):
    log_message('38;5;240', message)

def log_trace(message):
    log_message('38;5;046', message)

def log_command(args, color=None):
    string = bash_escape(args)
    if color is None:
        log_debug(string)
    else:
        log_message(color, string)
