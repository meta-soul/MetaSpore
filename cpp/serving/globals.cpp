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

#include <gflags/gflags.h>

namespace metaspore::serving {

DEFINE_uint64(compute_thread_num, 4UL, "Thread number for computing");
DEFINE_uint64(ort_intraop_thread_num, 4UL, "Thread number for computing");
DEFINE_uint64(ort_interop_thread_num, 4UL, "Thread number for computing");
DEFINE_uint64(background_thread_num, 2UL, "Thread number for background tasks");
DEFINE_uint64(grpc_server_threads, 4UL, "Thread number for grpc servers");
DEFINE_uint64(grpc_client_threads, 4UL, "Thread number for grpc clients");
DEFINE_string(grpc_listen_host, "0.0.0.0", "Listen host for grpc service");
DEFINE_string(grpc_listen_port, "50051", "Listen port for grpc service");
DEFINE_string(init_load_path, ".", "Load path to init during start");

} // namespace metaspore::serving
