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

#include <metaspore/feature_extraction_python_bindings.h>
#include <metaspore/io.h>
#include <metaspore/ms_ps_python_bindings.h>
#include <metaspore/model_metric_buffer.h>
#include <metaspore/ps_agent.h>
#include <metaspore/ps_runner.h>
#include <metaspore/pybind_utils.h>
#include <metaspore/tensor_store_python_bindings.h>

namespace py = pybind11;

PYBIND11_MODULE(_metaspore, m) {
    py::enum_<metaspore::NodeRole>(m, "NodeRole")
        .value("Coordinator", metaspore::NodeRole::Coordinator)
        .value("Server", metaspore::NodeRole::Server)
        .value("Worker", metaspore::NodeRole::Worker);

    py::class_<metaspore::NodeInfo>(m, "NodeInfo")
        .def_property("role", &metaspore::NodeInfo::GetRole, &metaspore::NodeInfo::SetRole)
        .def_property("node_id", &metaspore::NodeInfo::GetNodeId, &metaspore::NodeInfo::SetNodeId)
        .def_property("host_name", &metaspore::NodeInfo::GetHostName,
                      &metaspore::NodeInfo::SetHostName)
        .def_property("port", &metaspore::NodeInfo::GetPort, &metaspore::NodeInfo::SetPort)
        .def_property_readonly("address", &metaspore::NodeInfo::GetAddress)
        .def("__repr__", &metaspore::NodeInfo::ToString)
        .def("__str__", &metaspore::NodeInfo::ToShortString);

    py::class_<metaspore::ActorConfig, std::shared_ptr<metaspore::ActorConfig>>(m, "ActorConfig")
        .def(py::init<>())
        .def_property("agent_creator", &metaspore::ActorConfig::GetAgentCreator,
                      [](metaspore::ActorConfig &self, py::object creator) {
                          auto func = metaspore::make_shared_pyobject(creator);
                          self.SetAgentCreator([func] {
                              py::gil_scoped_acquire gil;
                              py::object obj = (*func)();
                              return metaspore::extract_shared_pyobject<metaspore::PSAgent>(obj);
                          });
                      })
        .def_property("agent_ready_callback", &metaspore::ActorConfig::GetAgentReadyCallback,
                      [](metaspore::ActorConfig &self, py::object cb) {
                          auto func = metaspore::make_shared_pyobject(cb);
                          self.SetAgentReadyCallback(
                              [func](std::shared_ptr<metaspore::PSAgent> agent) {
                                  py::gil_scoped_acquire gil;
                                  (*func)(agent);
                              });
                      })
        .def_property("transport_type", &metaspore::ActorConfig::GetTransportType,
                      &metaspore::ActorConfig::SetTransportType)
        .def_property("is_local_mode", &metaspore::ActorConfig::IsLocalMode,
                      &metaspore::ActorConfig::SetIsLocalMode)
        .def_property("use_kubernetes", &metaspore::ActorConfig::UseKubernetes,
                      &metaspore::ActorConfig::SetUseKubernetes)
        .def_property("root_uri", &metaspore::ActorConfig::GetRootUri,
                      &metaspore::ActorConfig::SetRootUri)
        .def_property("root_port", &metaspore::ActorConfig::GetRootPort,
                      &metaspore::ActorConfig::SetRootPort)
        .def_property("node_uri", &metaspore::ActorConfig::GetNodeUri,
                      &metaspore::ActorConfig::SetNodeUri)
        .def_property("node_interface", &metaspore::ActorConfig::GetNodeInterface,
                      &metaspore::ActorConfig::SetNodeInterface)
        .def_property("node_role", &metaspore::ActorConfig::GetNodeRole,
                      &metaspore::ActorConfig::SetNodeRole)
        .def_property("node_port", &metaspore::ActorConfig::GetNodePort,
                      &metaspore::ActorConfig::SetNodePort)
        .def_property(
            "this_node_info",
            [](const metaspore::ActorConfig &self) { return self.GetThisNodeInfo(); },
            &metaspore::ActorConfig::SetThisNodeInfo)
        .def_property_readonly("is_coordinator", &metaspore::ActorConfig::IsCoordinator)
        .def_property_readonly("is_server", &metaspore::ActorConfig::IsServer)
        .def_property_readonly("is_worker", &metaspore::ActorConfig::IsWorker)
        .def_property("bind_retry", &metaspore::ActorConfig::GetBindRetry,
                      &metaspore::ActorConfig::SetBindRetry)
        .def_property("heartbeat_interval", &metaspore::ActorConfig::GetHeartbeatInterval,
                      &metaspore::ActorConfig::SetHeartbeatInterval)
        .def_property("heartbeat_timeout", &metaspore::ActorConfig::GetHeartbeatTimeout,
                      &metaspore::ActorConfig::SetHeartbeatTimeout)
        .def_property("is_message_dumping_enabled",
                      &metaspore::ActorConfig::IsMessageDumpingEnabled,
                      &metaspore::ActorConfig::SetIsMessageDumpingEnabled)
        .def_property("is_resending_enabled", &metaspore::ActorConfig::IsResendingEnabled,
                      &metaspore::ActorConfig::SetIsResendingEnabled)
        .def_property("resending_timeout", &metaspore::ActorConfig::GetResendingTimeout,
                      &metaspore::ActorConfig::SetResendingTimeout)
        .def_property("resending_retry", &metaspore::ActorConfig::GetResendingRetry,
                      &metaspore::ActorConfig::SetResendingRetry)
        .def_property("drop_rate", &metaspore::ActorConfig::GetDropRate,
                      &metaspore::ActorConfig::SetDropRate)
        .def_property("server_count", &metaspore::ActorConfig::GetServerCount,
                      &metaspore::ActorConfig::SetServerCount)
        .def_property("worker_count", &metaspore::ActorConfig::GetWorkerCount,
                      &metaspore::ActorConfig::SetWorkerCount)
        .def("copy", &metaspore::ActorConfig::Copy);

    py::class_<metaspore::PSRunner>(m, "PSRunner")
        .def_static("run_ps", &metaspore::PSRunner::RunPS,
                    py::call_guard<py::gil_scoped_release>());

    py::class_<metaspore::SmartArray<uint8_t>>(m, "SmartArray", py::buffer_protocol())
        .def_buffer([](metaspore::SmartArray<uint8_t> &sa) {
            return py::buffer_info(sa.data(),       /* Pointer to buffer */
                                   sizeof(uint8_t), /* Size of one scalar */
                                   py::format_descriptor<uint8_t>::format(), /* Python struct-style
                                                                                format descriptor */
                                   1,                  /* Number of dimensions */
                                   {sa.size()},        /* Buffer dimensions */
                                   {sizeof(uint8_t)}); /* Strides (in bytes) for each index */
        });
    ;

    py::class_<metaspore::Message, std::shared_ptr<metaspore::Message>>(m, "Message")
        .def(py::init<>())
        .def_property_readonly(
            "message_id",
            [](const metaspore::Message &self) { return self.GetMessageMeta().GetMessageId(); })
        .def_property(
            "sender",
            [](const metaspore::Message &self) { return self.GetMessageMeta().GetSender(); },
            [](metaspore::Message &self, int value) { self.GetMessageMeta().SetSender(value); })
        .def_property(
            "receiver",
            [](const metaspore::Message &self) { return self.GetMessageMeta().GetReceiver(); },
            [](metaspore::Message &self, int value) { self.GetMessageMeta().SetReceiver(value); })
        .def_property(
            "is_request",
            [](const metaspore::Message &self) { return self.GetMessageMeta().IsRequest(); },
            [](metaspore::Message &self, bool value) { self.GetMessageMeta().SetIsRequest(value); })
        .def_property(
            "is_response",
            [](const metaspore::Message &self) { return !self.GetMessageMeta().IsRequest(); },
            [](metaspore::Message &self, bool value) {
                self.GetMessageMeta().SetIsRequest(!value);
            })
        .def_property(
            "is_exception",
            [](const metaspore::Message &self) { return self.GetMessageMeta().IsException(); },
            [](metaspore::Message &self, bool value) {
                self.GetMessageMeta().SetIsException(value);
            })
        .def_property(
            "body", [](const metaspore::Message &self) { return self.GetMessageMeta().GetBody(); },
            [](metaspore::Message &self, std::string value) {
                self.GetMessageMeta().SetBody(std::move(value));
            })
        .def_property_readonly(
            "slice_count", [](const metaspore::Message &self) { return self.GetSlices().size(); })
        .def("clear_slices", [](metaspore::Message &self) { self.ClearSlicesAndDataTypes(); })
        .def("get_slice",
             [](metaspore::Message &self, size_t index) {
                 metaspore::SmartArray<uint8_t> slice = self.GetSlice(index);
                 const metaspore::MessageMeta &meta = self.GetMessageMeta();
                 const metaspore::DataType type = meta.GetSliceDataTypes().at(index);
                 return metaspore::make_numpy_array(slice, type);
             })
#undef MS_DATA_TYPE_DEF
#define MS_DATA_TYPE_DEF(t, l, u)                                                                  \
    .def("add_slice", [](metaspore::Message &self, py::array_t<t> arr) {                           \
        auto obj = metaspore::make_shared_pyobject(arr);                                           \
        t *data = const_cast<t *>(arr.data(0));                                                    \
        auto sa = metaspore::SmartArray<t>::Create(data, arr.size(), [obj](t *) {});               \
        self.AddTypedSlice(sa);                                                                    \
    }) /**/
            MS_DATA_STRUCTURES_DATA_TYPES(MS_DATA_TYPE_DEF)
        .def("copy", &metaspore::Message::Copy)
        .def("__str__", &metaspore::Message::ToString);

    py::class_<metaspore::PSAgent, metaspore::PyPSAgent<>, std::shared_ptr<metaspore::PSAgent>>(
        m, "PSAgent")
        .def(py::init<>())
        .def("run", &metaspore::PSAgent::Run)
        .def("handle_request", &metaspore::PSAgent::HandleRequest)
        .def("finalize", &metaspore::PSAgent::Finalize)
        .def_property_readonly("is_coordinator", &metaspore::PSAgent::IsCoordinator)
        .def_property_readonly("is_server", &metaspore::PSAgent::IsServer)
        .def_property_readonly("is_worker", &metaspore::PSAgent::IsWorker)
        .def_property_readonly("server_count", &metaspore::PSAgent::GetServerCount)
        .def_property_readonly("worker_count", &metaspore::PSAgent::GetWorkerCount)
        .def_property_readonly("rank", &metaspore::PSAgent::GetAgentRank)
        .def("barrier", &metaspore::PSAgent::Barrier, py::call_guard<py::gil_scoped_release>())
        // This adds an overload to default to setting up a barrier among all workers.
        .def("barrier",
             [](metaspore::PSAgent &self) {
                 py::gil_scoped_release gil;
                 self.Barrier(metaspore::WorkerGroup);
             })
        .def("shutdown", &metaspore::PSAgent::Shutdown)
        .def("send_request",
             [](metaspore::PSAgent &self, metaspore::PSMessage req, py::object cb) {
                 auto func = metaspore::make_shared_pyobject(cb);
                 py::gil_scoped_release gil;
                 self.SendRequest(req, [func](metaspore::PSMessage req, metaspore::PSMessage res) {
                     py::gil_scoped_acquire gil;
                     (*func)(req, res);
                 });
             })
        .def("send_all_requests",
             [](metaspore::PSAgent &self, py::object reqs, py::object cb) {
                 auto func = metaspore::make_shared_pyobject(cb);
                 std::vector<metaspore::PSMessage> requests =
                     metaspore::make_cpp_vector<metaspore::PSMessage>(reqs);
                 py::gil_scoped_release gil;
                 self.SendAllRequests(std::move(requests),
                                      [func](std::vector<metaspore::PSMessage> reqs,
                                             std::vector<metaspore::PSMessage> ress) {
                                          py::gil_scoped_acquire gil;
                                          py::list requests = metaspore::make_python_list(reqs);
                                          py::list responses = metaspore::make_python_list(ress);
                                          (*func)(requests, responses);
                                      });
             })
        .def("broadcast_request",
             [](metaspore::PSAgent &self, metaspore::PSMessage req, py::object cb) {
                 auto func = metaspore::make_shared_pyobject(cb);
                 py::gil_scoped_release gil;
                 self.BroadcastRequest(
                     req, [func](metaspore::PSMessage req, std::vector<metaspore::PSMessage> ress) {
                         py::gil_scoped_acquire gil;
                         py::list responses = metaspore::make_python_list(ress);
                         (*func)(req, responses);
                     });
             })
        .def("send_response", &metaspore::PSAgent::SendResponse)
        .def("__str__", &metaspore::PSAgent::ToString);

    py::class_<metaspore::ModelMetricBuffer>(m, "ModelMetricBuffer")
        .def_static("update_buffer", &metaspore::ModelMetricBuffer::UpdateBuffer)
        .def_static("compute_auc", &metaspore::ModelMetricBuffer::ComputeAUC);

    py::class_<metaspore::InputStream, std::shared_ptr<metaspore::InputStream>>(m, "InputStream")
        .def(py::init<const std::string>())
        .def("read", [](metaspore::InputStream &stream, size_t size) {
            std::string buffer(size, '\0');
            const size_t nread = stream.Read(buffer.data(), size);
            buffer.resize(nread);
            return py::bytes(buffer);
        });

    py::class_<metaspore::OutputStream, std::shared_ptr<metaspore::OutputStream>>(m, "OutputStream")
        .def(py::init<const std::string>())
        .def("write", [](metaspore::OutputStream &stream, py::bytes data) {
            char *buffer;
            ssize_t length;
            if (PYBIND11_BYTES_AS_STRING_AND_SIZE(data.ptr(), &buffer, &length))
                py::pybind11_fail("Unable to extract bytes contents!");
            stream.Write(buffer, length);
        });

    m.def("stream_write_all",
          [](const std::string &url, py::bytes data) {
              char *buffer;
              ssize_t length;
              if (PYBIND11_BYTES_AS_STRING_AND_SIZE(data.ptr(), &buffer, &length))
                  py::pybind11_fail("Unable to extract bytes contents!");
              metaspore::StreamWriteAll(url, buffer, length);
          })
        .def("stream_write_all",
             [](const std::string &url, py::array data) {
                 const char *buffer = static_cast<const char *>(data.data());
                 const size_t length = data.nbytes();
                 metaspore::StreamWriteAll(url, buffer, length);
             })
        .def("stream_read_all",
             [](const std::string &url) {
                 std::string data = metaspore::StreamReadAll(url);
                 return py::bytes(data);
             })
        .def("stream_read_all",
             [](const std::string &url, py::array data) {
                 char *buffer = static_cast<char *>(data.mutable_data());
                 const size_t length = data.nbytes();
                 metaspore::StreamReadAll(url, buffer, length);
             })
        .def("ensure_local_directory", &metaspore::EnsureLocalDirectory)
        .def("get_metaspore_version", [] { return _METASPORE_VERSION; });

    metaspore::DefineTensorStoreBindings(m);
    metaspore::DefineFeatureExtractionBindings(m);
}
