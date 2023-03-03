import subprocess
import asyncio


async def _start_tracking_service():
    tracking_service_cmd = "python tracking_service.py "
    print("start tracking service:", tracking_service_cmd)
    subprocess.Popen(tracking_service_cmd, shell=True, env={
        "SERVICE_PORT": str(50001)
    })


async def _start_log_watcher():
    log_watcher_cmd = "python log_watcher.py "
    print("start log watcher:", log_watcher_cmd)
    subprocess.Popen(log_watcher_cmd, shell=True, env={
        "UPLOAD_TYPE": "S3",
        "UPLOAD_PATH": "s3://dmetasoul-test-bucket/jiangnan/workdir/",
        "ACCESS_KEY_ID": "AKIAR53IE6EWBKUCSTWG",
        "SECRET_ACCESS_KEY": "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",
        "ENDPOINT": "s3.cn-northwest-1.amazonaws.com.cn",    

    })


def main():
    asyncio.run(_start_tracking_service())
    asyncio.run(_start_log_watcher())
    subprocess.call(["tail", "-f", "/dev/null"])


main()
