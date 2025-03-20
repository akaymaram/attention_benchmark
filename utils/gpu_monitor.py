import subprocess
import torch

def monitor_GPU_usage():
    if not torch.cuda.is_available():
        print("No GPU available.")
        return None, None, None

    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,power.draw', '--format=csv,noheader,nounits'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        output = result.stdout.strip().split(',')
        gpu_percent = float(output[0])
        gpu_memory = float(output[1])
        gpu_power = float(output[2])
        return gpu_percent, gpu_memory, gpu_power
    except Exception as e:
        print(f"Error in monitoring GPU usage: {e}")
        return 0, 0, 0
