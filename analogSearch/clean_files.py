import os
import time
import shutil

def cleanup_old_files(directory, age_limit_days=3):
    """
    清理指定目录下超过指定天数的文件或目录
    """
    current_time = time.time()
    age_limit = age_limit_days * 24 * 60 * 60  # 秒

    for name in os.listdir(directory):
        path = os.path.join(directory, name)
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            # 无法获取时间就跳过
            continue

        age = current_time - mtime

        if os.path.isfile(path):
            if age > age_limit:
                os.remove(path)
                print(f"Deleted file: {path}")
        elif os.path.isdir(path):
            if age > age_limit:
                shutil.rmtree(path)
                print(f"Deleted directory: {path}")


def monitor_directory(directories, interval=60*60*3, age_limit_days=3):
    """
    监控指定目录，定期清理过期文件
    :param directory: 要监控的目录
    :param interval: 每次检查的间隔时间，单位秒，默认每3小时检查一次
    """
    if isinstance(directories, str):
        directories = [directories]
    # 启动前，确保每个目录至少存在一次
    for d in directories:
        os.makedirs(d, exist_ok=True)
    while True:
        for d in directories:
            cleanup_old_files(d, age_limit_days)
        time.sleep(interval)  # 每隔一段时间检查一次

# if __name__ == "__main__":
#     # 设置要监控的目录
#     tmp_directory = ["tmp/database_tmp", "tmp/result_csv_tmp"]
    
#     # 启动目录监控
#     monitor_directory(tmp_directory)
