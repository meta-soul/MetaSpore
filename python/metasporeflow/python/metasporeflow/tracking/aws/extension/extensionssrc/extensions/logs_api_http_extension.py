#!/bin/sh
''''exec python -u -- "$0" ${1+"$@"} # '''

import os
import sys
from pathlib import Path
from datetime import datetime
# Add lib folder to path to import boto3 library.
# Normally with Lambda Layers, python libraries are put into the /python folder which is in the path.
# As this extension is bringing its own Python runtime, and running a separate process, the path is not available.
# Hence, having the files in a different folder and adding it to the path, makes it available. 
lib_folder = Path(__file__).parent / "lib"
sys.path.insert(0,str(lib_folder))
import boto3

from logs_api_http_extension.http_listener import http_server_init, RECEIVER_PORT
from logs_api_http_extension.logs_api_client import LogsAPIClient
from logs_api_http_extension.extensions_api_client import ExtensionsAPIClient

from queue import Queue
from pymongo import MongoClient
import json

"""Here is the sample extension code.
    - The extension runs two threads. The "main" thread, will register with the Extension API and process its invoke and
    shutdown events (see next call). The second "listener" thread listens for HTTP POST events that deliver log batches.
    - The "listener" thread places every log batch it receives in a synchronized queue; during each execution slice,
    the "main" thread will make sure to process any event in the queue before returning control by invoking next again.
    - Note that because of the asynchronous nature of the system, it is possible that logs for one invoke are
    processed during the next invoke slice. Likewise, it is possible that logs for the last invoke are processed during
    the SHUTDOWN event.

Note: 

1.  This is a simple example extension to help you understand the Lambda Logs API.
    This code is not production ready. Use it with your own discretion after testing it thoroughly.  

2.  The extension code starts with a shebang. This is to bring Python runtime to the execution environment.
    This works if the lambda function is a python3.x function, therefore it brings the python3.x runtime with itself.
    It may not work for python 2.7 or other runtimes. 
    The recommended best practice is to compile your extension into an executable binary and not rely on the runtime.
  
3.  This file needs to be executable, so make sure you add execute permission to the file 
    `chmod +x logs_api_http_extension.py`

"""

class LogsAPIHTTPExtension():
    def __init__(self, agent_name, registration_body, subscription_body):
 #       print(f"extension.logs_api_http_extension: Initializing LogsAPIExternalExtension {agent_name}")
        self.agent_name = agent_name
        self.queue = Queue()
        self.logs_api_client = LogsAPIClient()
        self.extensions_api_client = ExtensionsAPIClient()

        # Register early so Runtime could start in parallel
        self.agent_id = self.extensions_api_client.register(self.agent_name, registration_body)

        # Start listening before Logs API registration
#        print(f"extension.logs_api_http_extension: Starting HTTP Server {agent_name}")
        http_server_init(self.queue)
        self.logs_api_client.subscribe(self.agent_id, subscription_body)

        # initialize mongo client
        self._mongo_client = self._get_mongo_client()

    def _get_mongo_client(self):
        client = MongoClient(os.environ['METASPOREFLOW_TRACKING_DB_URI'])
        mongo_client = client[os.environ['METASPOREFLOW_TRACKING_DB_DATABASE']]
        return mongo_client

    def _save_collections_to_mongodb(self, record):
        collection_table = self._mongo_client[os.environ['METASPOREFLOW_TRACKING_DB_TABLE']]
        collection_table.insert_one(eval(record))

    def run_forever(self):
        # Configuring S3 Connection
        s3_bucket = (os.environ['S3_BUCKET_NAME'])
        s3 = boto3.resource('s3')
        print(f"extension.logs_api_http_extension: Receiving Logs {self.agent_name}")

        while True:
            self.extensions_api_client.next(self.agent_id)
            # Process the received batches if any.
            while not self.queue.empty():
                batch = self.queue.get_nowait()
                s3_filename = (os.environ['AWS_LAMBDA_FUNCTION_NAME'])+'-'+(datetime.now().strftime('%Y-%m-%d-%H:%M:%S.%f'))+'.log'
                try:
                    # Save to S3
                    s3.Bucket(s3_bucket).put_object(Key=s3_filename, Body=str(batch))
                    # Save to MongoDB
                    if os.environ['METASPOREFLOW_TRACKING_DB_ENABLE'] == 'true' and os.environ['METASPOREFLOW_TRACKING_DB_TYPE'] == 'mongodb':
                        for item in range(len(batch)):
                            record_str = batch[item]['record']
                            record = json.loads(record_str)
                            self._save_collections_to_mongodb(record)
                except Exception as e:
                    raise Exception(f"Error sending log to S3 {e}") from e
_REGISTRATION_BODY = {
    "events": ["INVOKE", "SHUTDOWN"],
}

# Subscribe to platform logs and receive them on ${local_ip}:4243 via HTTP protocol.

TIMEOUT_MS = 1000 # Maximum time (in milliseconds) that a batch is buffered.
MAX_BYTES = 262144 # Maximum size in bytes that the logs are buffered in memory.
MAX_ITEMS = 10000 # Maximum number of events that are buffered in memory.

_SUBSCRIPTION_BODY = {
    "destination":{
        "protocol": "HTTP",
        "URI": f"http://sandbox:{RECEIVER_PORT}",
    },
    "types": ["function"],
    "buffering": {
        "timeoutMs": int(os.environ['METASPOREFLOW_TRACKING_LOG_BUFFER_TIMEOUT_MS']),
        "maxBytes": int(os.environ['METASPOREFLOW_TRACKING_LOG_BUFFER_MAX_BYTES']),
        "maxItems": int(os.environ['METASPOREFLOW_TRACKING_LOG_BUFFER_MAX_ITEMS'])
    }
}

def main():
#    print(f"extension.logs_api_http_extension: Starting Extension {_REGISTRATION_BODY} {_SUBSCRIPTION_BODY}")
    # Note: Agent name has to be file name to register as an external extension
    ext = LogsAPIHTTPExtension(os.path.basename(__file__), _REGISTRATION_BODY, _SUBSCRIPTION_BODY)
    ext.run_forever()

if __name__ == "__main__":
    main()
