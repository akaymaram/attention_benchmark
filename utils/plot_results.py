import matplotlib.pyplot as plt
import os
import cv2
from google.colab.patches import cv2_imshow
import numpy as np

def plot_result(results_dict, save_path="plots"):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    grouped_results = {}
    for key, value in results_dict.items():
        model_name = key[0]
        metric_name = key[1]
        unit = key[2]

        if (metric_name, unit) not in grouped_results:
            grouped_results[(metric_name, unit)] = {}
        grouped_results[(metric_name, unit)][model_name] = value

    for (metric_name, unit), models_data in grouped_results.items():
        if metric_name in ["FLOPS", "gpu_memory", "model_size"]:
            attention_names = []
            values = []

            for model_name, data in models_data.items():
                wrapped_label = "\n".join(model_name.split())
                attention_names.append(wrapped_label)

                if metric_name == "FLOPS":
                    avg_value = data["FLOPS"].mean()
                elif metric_name == "model_size":
                    avg_value = data["Model size"].values[0]
                else:
                    avg_value = data.iloc[:, 1].mean()
                values.append(avg_value)

            plt.figure(figsize=(12, 8))
            plt.bar(attention_names, values, color="skyblue")
            plt.title(f"Comparison of {metric_name}", fontsize=16)
            plt.xlabel("Attention Mechanism", fontsize=14)
            plt.ylabel(f"{metric_name} ({unit})", fontsize=14)
            plt.grid(axis="y")

            plt.xticks(rotation=60, ha="right", fontsize=12)

            filename = f"{metric_name.replace(' ', '_')}_{unit}_bar.png"
            filepath = os.path.join(save_path, filename)
            plt.savefig(filepath, bbox_inches="tight")
            plt.close()

            image = cv2.imread(filepath)
            if image is not None:
                cv2_imshow(image)
            else:
                print(f"Failed to load image: {filepath}")
        else:
            plt.figure(figsize=(12, 8))

            for model_name, data in models_data.items():
                x = data["Epoch"]
                y = data.iloc[:, 1]
                plt.plot(x, y, marker='o', linestyle='-', label=model_name)

            plt.title(f"Comparison of {metric_name}", fontsize=16)
            plt.xlabel("Epoch", fontsize=14)
            plt.ylabel(f"{metric_name} ({unit})", fontsize=14)
            plt.grid(True)
            plt.legend(fontsize=12)

            filename = f"{metric_name.replace(' ', '_')}_{unit}.png"
            filepath = os.path.join(save_path, filename)
            plt.savefig(filepath, bbox_inches="tight")
            plt.close()

            image = cv2.imread(filepath)
            if image is not None:
                cv2_imshow(image)
            else:
                print(f"Failed to load image: {filepath}")
