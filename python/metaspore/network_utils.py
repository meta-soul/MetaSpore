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

def get_host_ip():
    import socket
    host_name = socket.gethostname()
    host_ip = socket.gethostbyname(host_name)
    return host_ip

def get_available_endpoint():
    import socket
    import random
    host_name = socket.gethostname()
    host_ip = socket.gethostbyname(host_name)
    addr_info = socket.getaddrinfo(host_ip, None)
    ip_family = addr_info[0][0]
    with socket.socket(ip_family, socket.SOCK_STREAM) as sock:
        try:
            sock.bind(('', 0))
            _, port = sock.getsockname()
            return host_ip, port
        except socket.error as e:
            message = "can not find bindable port "
            message += "on host %s(%s)" % (host_name, host_ip)
            raise RuntimeError(message) from e
