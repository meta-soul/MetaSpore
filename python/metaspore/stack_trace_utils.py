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

import os
import threading
import ctypes

def gettid():
    SYS_gettid = 186
    libc = ctypes.cdll.LoadLibrary('libc.so.6')
    tid = libc.syscall(SYS_gettid)
    return tid

def get_thread_identifier():
    string = 'pid: %d, ' % os.getpid()
    string += 'tid: %d, ' % gettid()
    string += 'thread: 0x%x' % threading.current_thread().ident
    return string
