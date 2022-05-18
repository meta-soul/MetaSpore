#!/bin/bash
set -x
consul watch -http-addr=http://${CONSUL_IP}:8500 -type keyprefix -prefix ${CONSUL_KEY_PREFIX} curl -H 'Content-Type:application/json' -X POST --data-binary @- http://${POD_IP}:8080/notify