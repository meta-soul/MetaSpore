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

# Demonstrates code to register as an extension.

LAMBDA_AGENT_NAME_HEADER_KEY = "Lambda-Extension-Name"
LAMBDA_AGENT_IDENTIFIER_HEADER_KEY = "Lambda-Extension-Identifier"

class ExtensionsAPIClient():
    def __init__(self):
        try:
            runtime_api_address = os.environ['AWS_LAMBDA_RUNTIME_API']
            self.runtime_api_base_url = f"http://{runtime_api_address}/2020-01-01/extension"
        except Exception as e:
            raise Exception(f"AWS_LAMBDA_RUNTIME_API is not set {e}") from e

    # Register as early as possible - the runtime initialization starts after all extensions have registered.
    def register(self, agent_unique_name, registration_body):
        try:
            print(f"extension.extensions_api_client: Registering Extension at ExtensionsAPI address: {self.runtime_api_base_url}")
            req = urllib.request.Request(f"{self.runtime_api_base_url}/register")
            req.method = 'POST'
            req.add_header(LAMBDA_AGENT_NAME_HEADER_KEY, agent_unique_name)
            req.add_header("Content-Type", "application/json")
            data = json.dumps(registration_body).encode("utf-8")
            req.data = data
            resp = urllib.request.urlopen(req)
            if resp.status != 200:
                print(f"extension.extensions_api_client: /register request to ExtensionsAPI failed. Status:  {resp.status}, Response: {resp.read()}")
                # Fail the extension
                sys.exit(1)
            agent_identifier = resp.headers.get(LAMBDA_AGENT_IDENTIFIER_HEADER_KEY)
#            print(f"extension.extensions_api_client: received agent_identifier header  {agent_identifier}")
            return agent_identifier
        except Exception as e:
            raise Exception(f"Failed to register to ExtensionsAPI: on {self.runtime_api_base_url}/register \
                with agent_unique_name:{agent_unique_name}  \
                and registration_body:{registration_body}\nError: {e}") from e

    # Call the following method when the extension is ready to receive the next invocation
    # and there is no job it needs to execute beforehand.
    def next(self, agent_id):
        try:
            print(f"extension.extensions_api_client: Requesting /event/next from Extensions API")
            req = urllib.request.Request(f"{self.runtime_api_base_url}/event/next")
            req.method = 'GET'
            req.add_header(LAMBDA_AGENT_IDENTIFIER_HEADER_KEY, agent_id)
            req.add_header("Content-Type", "application/json")
            resp = urllib.request.urlopen(req)
            if resp.status != 200:
                print(f"extension.extensions_api_client: /event/next request to ExtensionsAPI failed. Status: {resp.status}, Response: {resp.read()} ")
                # Fail the extension
                sys.exit(1)
            data = resp.read()
            print(f"extension.extensions_api_client:  Received event from ExtensionsAPI: {data}")
            return data
        except Exception as e:
            raise Exception(f"Failed to get /event/next from ExtensionsAPI: {e}") from e
