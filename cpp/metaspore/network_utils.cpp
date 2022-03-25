//
// Copyright 2022 DMetaSoul
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//

#include <arpa/inet.h>
#include <ifaddrs.h>
#include <memory>
#include <metaspore/network_utils.h>
#include <net/if.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

namespace metaspore::network_utils {

std::string get_ip(const std::string &interface) {
    struct ifaddrs *ifas = nullptr;
    getifaddrs(&ifas);
    std::unique_ptr<struct ifaddrs, decltype(&freeifaddrs)> ifas_guard(ifas, &freeifaddrs);
    for (auto ifa = ifas; ifa != nullptr; ifa = ifa->ifa_next) {
        if (ifa->ifa_addr == nullptr)
            continue;
        if (ifa->ifa_addr->sa_family != AF_INET)
            continue;
        if (interface != ifa->ifa_name)
            continue;
        auto addr = reinterpret_cast<struct sockaddr_in *>(ifa->ifa_addr);
        void *temp_addr_ptr = &addr->sin_addr;
        char address_buffer[INET_ADDRSTRLEN];
        inet_ntop(AF_INET, temp_addr_ptr, address_buffer, INET_ADDRSTRLEN);
        return address_buffer;
    }
    return {};
}

std::string get_interface_and_ip(std::string &interface) {
    struct ifaddrs *ifas = nullptr;
    getifaddrs(&ifas);
    std::unique_ptr<struct ifaddrs, decltype(&freeifaddrs)> ifas_guard(ifas, &freeifaddrs);
    for (auto ifa = ifas; ifa != nullptr; ifa = ifa->ifa_next) {
        if (ifa->ifa_addr == nullptr)
            continue;
        if (ifa->ifa_addr->sa_family != AF_INET)
            continue;
        if (ifa->ifa_flags & IFF_LOOPBACK)
            continue;
        auto addr = reinterpret_cast<struct sockaddr_in *>(ifa->ifa_addr);
        void *temp_addr_ptr = &addr->sin_addr;
        char address_buffer[INET_ADDRSTRLEN];
        inet_ntop(AF_INET, temp_addr_ptr, address_buffer, INET_ADDRSTRLEN);
        interface = ifa->ifa_name;
        return address_buffer;
    }
    return {};
}

int get_available_port() {
    struct sockaddr_in addr;
    addr.sin_port = htons(0);
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
    const int sock = socket(AF_INET, SOCK_STREAM, 0);
    if (bind(sock, (struct sockaddr *)&addr, sizeof(struct sockaddr_in)) != 0) {
        perror("bind():");
        return 0;
    }
    socklen_t addr_len = sizeof(struct sockaddr_in);
    if (getsockname(sock, (struct sockaddr *)&addr, &addr_len) != 0) {
        perror("getsockname():");
        return 0;
    }
    const int port = ntohs(addr.sin_port);
    close(sock);
    return port;
}

} // namespace metaspore::network_utils
