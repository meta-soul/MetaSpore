/bin/sh
set -x
consul watch -http-addr=http://${CONSUL_IP}:8500 -type keyprefix -prefix dev/ curl -H 'Content-Type:application/json' -X POST --data-binary @- http://${POD_IP}:8080/notify