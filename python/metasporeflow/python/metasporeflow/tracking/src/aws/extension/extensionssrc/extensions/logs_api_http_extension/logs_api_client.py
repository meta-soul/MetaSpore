#  Copyright 2023 DMetaSoul
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import json
import os
import sys
import urllib.request

# Demonstrates code to call the Logs API to subscribe to log events

LAMBDA_AGENT_IDENTIFIER_HEADER_KEY = "Lambda-Extension-Identifier"

class LogsAPIClient:
    def __init__(self):
        try:
            runtime_api_address = os.environ['AWS_LAMBDA_RUNTIME_API']
            self.logs_api_base_url = f"http://{runtime_api_address}/2020-08-15/logs"
        except Exception as e:
            raise Exception(f"AWS_LAMBDA_RUNTIME_API is not set {e}") from e

    # Method to call the Logs API to subscribe to log events.
    def subscribe(self, agent_id, subscription_body):
        try:
            print(f"extension.logs_api_client: Subscribing to Logs API on {self.logs_api_base_url}")
            req = urllib.request.Request(f"{self.logs_api_base_url}")
            req.method = 'PUT'
            req.add_header(LAMBDA_AGENT_IDENTIFIER_HEADER_KEY, agent_id)
            req.add_header("Content-Type", "application/json")
            data = json.dumps(subscription_body).encode("utf-8")
            req.data = data
            resp = urllib.request.urlopen(req)
            if resp.status != 200:
                print(f"extension.logs_api_client: Could not subscribe to Logs API: {resp.status} {resp.read()}")
                # Fail the extension
                sys.exit(1)
#            print(f"extension.logs_api_client: Succesfully subscribed to Logs API: {resp.read()}")
        except Exception as e:
            raise Exception(f"Failed to subscribe to Logs API on {self.logs_api_base_url} with id: {agent_id} \
                and subscription_body: {json.dumps(subscription_body).encode('utf-8')} \nError:{e}") from e