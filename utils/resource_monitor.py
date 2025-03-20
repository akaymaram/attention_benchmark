import psutil

def monitor_resource_usage():
    memory_info = psutil.virtual_memory()
    cpu_percent = psutil.cpu_percent(interval=0)
    used_memory_gb = memory_info.used / (1024 ** 3)
    total_memory_gb = memory_info.total / (1024 ** 3)
    return used_memory_gb, total_memory_gb, cpu_percent
