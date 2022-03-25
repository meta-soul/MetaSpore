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

#include <cxxabi.h>
#include <execinfo.h>
#include <metaspore/stack_trace_utils.h>
#include <regex>
#include <sstream>
#include <stdint.h>
#include <vector>

namespace metaspore {

namespace {

std::vector<void *> GetStackTraceAddresses() {
    std::vector<void *> vec(1024);
    for (;;) {
        const int size = backtrace(&vec[0], (int)vec.size());
        if (size == vec.size())
            vec.resize(vec.size() * 2);
        else {
            vec.resize(size);
            return vec;
        }
    }
}

std::vector<std::string> GetStackTraceSymbols() {
    std::vector<void *> addresses = GetStackTraceAddresses();
    char **symbols = backtrace_symbols(&addresses[0], (int)addresses.size());
    std::unique_ptr<char *, decltype(&free)> symbols_guard(symbols, &free);
    std::vector<std::string> result;
    if (symbols)
        result.insert(result.end(), symbols, symbols + addresses.size());
    return result;
}

bool DecodeStackTraceSymbol(const std::string &symbol, std::string &file_name,
                            std::string &function_name, uintptr_t &offset, uintptr_t &address) {
    static const std::regex re("(.+)\\(([^+]+)\\+0x([0-9a-f]+)\\) \\[0x([0-9a-f]+)\\]");
    std::smatch m;
    if (!std::regex_match(symbol, m, re))
        return false;
    file_name = m[1].str();
    function_name = m[2].str();
    offset = std::stoull(m[3].str(), nullptr, 16);
    address = std::stoull(m[4].str(), nullptr, 16);
    int status = 0;
    char *demangled = abi::__cxa_demangle(function_name.c_str(), NULL, NULL, &status);
    std::unique_ptr<char, decltype(&free)> demangled_guard(demangled, &free);
    if (!demangled)
        return false;
    function_name = demangled;
    return true;
}

std::string DemangleStackTraceSymbol(const std::string &symbol) {
    std::string file_name;
    std::string function_name;
    uintptr_t offset;
    uintptr_t address;
    if (!DecodeStackTraceSymbol(symbol, file_name, function_name, offset, address))
        return symbol;
    std::ostringstream sout;
    sout << file_name << "(" << function_name << "+0x" << std::hex << offset << ")";
    sout << " [0x" << address << "]";
    return sout.str();
}

} // namespace

std::string GetStackTrace() {
    const int offset = 3;
    std::ostringstream sout;
    std::vector<std::string> symbols = GetStackTraceSymbols();
    sout << "Stack trace returned " << (symbols.size() - offset) << " entries:";
    for (size_t i = offset; i < symbols.size(); i++)
        sout << "\n[bt] (" << (i - offset) << ") " << DemangleStackTraceSymbol(symbols[i]);
    return sout.str();
}

} // namespace metaspore
