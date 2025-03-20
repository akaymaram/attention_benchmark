from numpy import False_
import torch
import torch.optim as optim
from transformers import GPT2Tokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import AutoModelForCausalLM, AutoConfig
from data.dataset import TextDataset

from utils.resource_monitor import monitor_resource_usage
from utils.gpu_monitor import monitor_GPU_usage
from utils.flops_counter import measure_flops
from utils.plot_results import plot_result
from config import (
    only_use_single_attention,
    use_flops,
    only_use_first_serveral_batch,
    number_first_serveral_batch,
    batch_size_number,
    max_epoch,
    fast_test,
    full_test,
    model_name,
)
from attention_modules import GroupQueryAttention,ScaledDotProductAttention, MultiHeadFlexAttention, SparseFlexAttention, LinearFlexAttention, LSHAttention, SlidingWindowAttention, FlashAttention,BaselineAttention

from torch.utils.data import DataLoader
from datasets import load_dataset
import pandas as pd
import os
import time


import torch
import torch.nn as nn
import torch.optim as optim
from transformers import GPT2Tokenizer, GPT2LMHeadModel, LlamaTokenizer, LlamaForCausalLM
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
import psutil
import time
from fvcore.nn import FlopCountAnalysis
import logging
import os
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
from flash_attn.flash_attn_interface import flash_attn_func
import subprocess

import pickle

from models.gpt2_custom import GPT2CustomAttentionModel
#from models.smolLM2_custom import SmolLM2CustomAttentionModel
#from models.llama_custom import LlamaCustomAttentionModel

# Load model directly
#from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import GPT2Tokenizer, GPT2LMHeadModel

from datetime import datetime



#from load_data_model import load_data_and_model
start_time = time.perf_counter()

# Load tokenizer and dataset
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
config = GPT2LMHeadModel.from_pretrained(model_name).config



#config = GPT2CustomAttentionModel.from_pretrained('gpt2').config
dataset = load_dataset("allenai/tulu-v2-sft-mixture", split='train')
text_dataset = TextDataset(dataset, tokenizer)
dataloader = DataLoader(text_dataset, batch_size=batch_size_number, shuffle=True)


def train_model(model, dataloader, optimizer, criterion, device, num_epochs):
    cpu_usage_data = []  # List to accumulate data for DataFrame
    memory_usage_data = []
    flops_data = []
    gpu_percent_data = []
    gpu_memory_data = []
    gpu_power_data = []
    disk_io_read_data = []
    disk_io_write_data = []
    inference_time_data = []
    training_time_data = []
    loss_data = []


    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        model.train()

        start_time = time.time()
        num_batch = 0

        total_mem_usage = 0
        total_cpu_usage = 0
        total_flops = 0
        total_gpu_percent = 0
        total_gpu_memory = 0
        total_gpu_power = 0
        disk_io_read_start = psutil.disk_io_counters().read_bytes
        disk_io_write_start = psutil.disk_io_counters().write_bytes
        total_inference_time = 0
        total_training_time = 0
        total_loss = 0


        for batch in dataloader:
            if num_batch == number_first_serveral_batch and only_use_first_serveral_batch:
                break
            num_batch += 1
            inputs, masks = batch
            inputs, masks = inputs.to(device), masks.to(device)

            # Skip batch if any input ID has zero length
            if inputs.shape[1] == 0:
                logging.debug("Skipping batch with empty input IDs.")
                continue

            # Move inputs and masks to the correct device
            inputs, masks = inputs.to(device), masks.to(device)

            # Track inference time
            optimizer.zero_grad()
            inference_start = time.time()
            outputs = model(input_ids=inputs, attention_mask=masks, labels=inputs)
            total_inference_time+=time.time() - inference_start



            # Extract the loss from the outputs tuple
            loss = outputs[0]
            loss.backward()
            #the next line can be commented
            #torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.5)  


            optimizer.step()

            total_loss += loss.item()

            #print(f"Loss: {loss.item()}")
            # Measure and log resource usage
            mem_usage,mem_total, cpu_usage = monitor_resource_usage()
            gpu_percent_usage ,gpu_memory_usage ,gpu_power_usage = monitor_GPU_usage()
            total_mem_usage += mem_usage
            total_cpu_usage += cpu_usage
            flops = 1
            if use_flops:
                total_flops += measure_flops(model, inputs)
            total_gpu_percent += gpu_percent_usage
            total_gpu_memory += gpu_memory_usage
            total_gpu_power += gpu_power_usage


        memory_average_usage = total_mem_usage / num_batch
        cpu_average_usage = total_cpu_usage / num_batch
        flops_average = total_flops / num_batch
        gpu_percent_average = total_gpu_percent / num_batch
        gpu_memory_average = total_gpu_memory / num_batch
        gpu_power_average = total_gpu_power / num_batch
        disk_io_read_end = psutil.disk_io_counters().read_bytes
        disk_io_write_end = psutil.disk_io_counters().write_bytes
        #average_inference_time = total_inference_time / num_batch
        #we do not need average, just use total inference time for all batches?
        total_training_time = time.time() - start_time
        loss_average = total_loss / num_batch


        cpu_usage_data.append({'Epoch': epoch+1, 'CPU_Usage': cpu_average_usage})
        memory_usage_data.append({'Epoch': epoch+1, 'Memory_Usage': memory_average_usage})
        flops_data.append({'Epoch': epoch+1, 'FLOPS': flops_average})
        gpu_percent_data.append({'Epoch': epoch+1, 'GPU_Utilization_Percentage': gpu_percent_average})
        gpu_memory_data.append({'Epoch': epoch+1, 'GPU_Memory': gpu_memory_average})
        gpu_power_data.append({'Epoch': epoch+1, 'GPU_Power': gpu_power_average})
        disk_io_read_data.append({'Epoch': epoch+1, 'Disk_IO_Read': disk_io_read_end - disk_io_read_start})
        disk_io_write_data.append({'Epoch': epoch+1, 'Disk_IO_Write': disk_io_write_end - disk_io_write_start})
        inference_time_data.append({'Epoch': epoch+1, 'Inference_Time': total_inference_time})
        training_time_data.append({'Epoch': epoch+1, 'Training_Time': total_training_time})
        loss_data.append({'Epoch': epoch+1, 'Loss': loss_average})







    cpu_usage_df = pd.DataFrame(cpu_usage_data)
    memory_usage_df = pd.DataFrame(memory_usage_data)
    flops_df = pd.DataFrame(flops_data)
    gpu_percent_df = pd.DataFrame(gpu_percent_data)
    gpu_memory_df = pd.DataFrame(gpu_memory_data)
    gpu_power_df = pd.DataFrame(gpu_power_data)
    disk_io_read_df = pd.DataFrame(disk_io_read_data)
    disk_io_write_df = pd.DataFrame(disk_io_write_data)
    inference_time_df = pd.DataFrame(inference_time_data)
    training_time_df = pd.DataFrame(training_time_data)
    loss_data = pd.DataFrame(loss_data)



    return cpu_usage_df, memory_usage_df, flops_df, gpu_percent_df, gpu_memory_df, gpu_power_df, disk_io_read_df, disk_io_write_df, inference_time_df, training_time_df,loss_data



# Define attention modules to use
attention_modules = [
    
    ScaledDotProductAttention,
    BaselineAttention,
    MultiHeadFlexAttention,
    SparseFlexAttention,
    LinearFlexAttention,
    LSHAttention,
    SlidingWindowAttention,
    GroupQueryAttention,
    FlashAttention
]

if only_use_single_attention:
	attention_modules = [
    FlashAttention,
    BaselineAttention,

  ]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = torch.nn.CrossEntropyLoss()
results_dict = {}


# Training loop
for attention_module in attention_modules:
    print(f"Training with {attention_module.__name__}...")
    #model = LlamaCustomAttentionModel(config, attention_module).to(device)
    model = GPT2CustomAttentionModel(config, attention_module).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=5e-5)



    cpu_usage_per_epoch, memory_usage_per_epoch, flops_per_epoch, gpu_percent_per_epoch, \
    gpu_memory_per_epoch, gpu_power_per_epoch, disk_io_read_per_epoch, disk_io_write_per_epoch, \
    inference_time_per_epoch, training_time_per_epoch, loss_per_epoch = train_model(
        model, dataloader, optimizer, criterion, device, max_epoch
    )

    torch.save(model.state_dict(), f'{attention_module.__name__}.pth')
    model_size = os.path.getsize(f'{attention_module.__name__}.pth') / (1024 ** 2)  # Convert to MB
    model_size_df = pd.DataFrame({"Epoch": "n", "Model size": [model_size]})
    os.remove(f'{attention_module.__name__}.pth')

    # Store results in dictionary
    results_dict[(attention_module.__name__, "cpu_usage", "percent")] = cpu_usage_per_epoch
    results_dict[(attention_module.__name__, "memory_usage", "GB")] = memory_usage_per_epoch
    results_dict[(attention_module.__name__, "FLOPS", "FLOPS")] = flops_per_epoch
    results_dict[(attention_module.__name__, "gpu_utilization_percentage", "percent")] = gpu_percent_per_epoch
    results_dict[(attention_module.__name__, "gpu_memory", "MB")] = gpu_memory_per_epoch
    results_dict[(attention_module.__name__, "gpu_power", "W")] = gpu_power_per_epoch
    results_dict[(attention_module.__name__, "disk_io_read", "MB")] = disk_io_read_per_epoch
    results_dict[(attention_module.__name__, "disk_io_write", "MB")] = disk_io_write_per_epoch
    results_dict[(attention_module.__name__, "model_size", "MB")] = model_size_df
    results_dict[(attention_module.__name__, "inference_time", "s")] = inference_time_per_epoch
    results_dict[(attention_module.__name__, "training_time", "s")] = training_time_per_epoch
    results_dict[(attention_module.__name__, "loss", "loss")] = loss_per_epoch


# Print current module information
print(f"Information for module '{attention_module.__name__}':")
for key, value in results_dict.items():
    print(f"{key[1]} ({key[2]}): {value}")

test_type="normal_test"
if  fast_test or  full_test:
    test_type = "full_test" if full_test else "fast_test"

basic_info = f'{datetime.now().strftime("%Y-%m-%d,%H:%M:%S")},{torch.cuda.get_device_name(0)},{test_type}_final_results_{model_name}_batchNumber{number_first_serveral_batch}_batchSize{batch_size_number}'

plot_result(results_dict,'results/'+basic_info)

with open(f"results/{basic_info}/final_results_{model_name}_batchNumber{number_first_serveral_batch}_batchSize{batch_size_number}.pkl", "wb") as f:
    pickle.dump(results_dict, f)

# Print configuration
print("Configuration:")
print(f"only_use_first_serveral_batch = {only_use_first_serveral_batch}")
print(f"number_first_serveral_batch = {number_first_serveral_batch}")
print(f"batch_size_number = {batch_size_number}")
print(f"only_use_single_attention = {only_use_single_attention}")
print(f"use_flops = {use_flops}")
print(f"max_epoch = {max_epoch}")
print(f"fast_test = {fast_test}")

end_time = time.perf_counter()
total_time = end_time - start_time
print(f"Total run time: {total_time:.2f} seconds")