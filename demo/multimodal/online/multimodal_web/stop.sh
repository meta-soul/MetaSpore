ps aux | grep 'flask' | grep 'port 8090' | awk -F" " '{print $2}' | xargs kill -9
