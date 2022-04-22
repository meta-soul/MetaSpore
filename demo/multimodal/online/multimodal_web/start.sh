export FLASK_APP=app
nohup flask run --host=127.0.0.1 --port 8090 > app.log 2>&1 &
