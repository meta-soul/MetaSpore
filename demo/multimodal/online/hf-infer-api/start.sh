PORT=8098
if [ -z "$PORT" ]; then
    echo "usage: sh start.sh <port>"
    exit
fi
if [ ! -f ".env" ]; then
    echo "Please create a .env file"
    exit
fi

export $(grep -v '^#' .env | xargs)
lsof -t -i:${PORT} | xargs kill -9 > /dev/null 2>&1
nohup uvicorn main:app --host=127.0.0.1 --port=${PORT} > start.log 2>&1 &
