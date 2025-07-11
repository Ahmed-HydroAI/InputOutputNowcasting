import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from prepare_sequences import extract_sequences
from metrics import calculate_MAE, calculate_RMSE, CSI
from utils import (
    process_data,
    extract_inputs,
    extract_labels,
    data_preprocessing,
    data_postprocessing,
    load_model,
    make_predictions,
    plot_animations,
    plot_predictions_frame,
)

# ------------------------
# Configuration
# ------------------------

# define the data directory
data_dir = ""

# 1  → Each model predicts 1 frame recursively for a total of 12 frames
# 6  → Each model predicts 6 frames, then recursively predicts the remaining 6
# 12 → Each model predicts all 12 frames in a single forward pass
n_output_frames = 1  # Choose from 1, 6, or 12

# directory to model weights
checkpoint_dir = ""

#directory to save outputs such as plots and metrics results
output_dir = ""
os.makedirs(output_dir, exist_ok=True)

# select the requires input configuration
# Input configurations (number of past frames used)

model_config = {
    2: {"slice": slice(22, 24), "color": 'blue'},
    3: {"slice": slice(21, 24), "color": 'red'},
    4: {"slice": slice(20, 24), "color": 'green'},
    6: {"slice": slice(18, 24), "color": 'cyan'},
    8: {"slice": slice(16, 24), "color": 'grey'},
    10: {"slice": slice(14, 24), "color": 'orange'},
    12: {"slice": slice(12, 24), "color": 'teal'}
}

selected_models = [f"Model_{k}" for k in model_config.keys()]
model_rename_dict = {f"Model_{k}": f"Input_{k}" for k in model_config.keys()}

#select the lead times required for plotting 
lead_times = [0, 5, 11]  # Indices to plot (0 = t+5min)

#select thresholded required for thersholded metrics calculation
thresholds = [0.1, 1.0, 3.0, 5.0, 8.0, 10.0]

# ------------------------
# Data Preparation
# ------------------------
all_files = sorted(os.listdir(data_dir))

#change the charecteristics of the generating sequences as required
sequences = extract_sequences(data_dir, all_files, num_frames=36, window=36, min_nonzero_ratio=0.5, time_interval=5)
sequences = sequences[:5]  # Limit number of sequences for testing, change as required
total_lead_frames = 12

print(f"Extracted {len(sequences)} valid sequences")

# ------------------------
# Evaluation
# ------------------------
results = {}
model_instances = {}
metrics_all = {"MAE": {}, "RMSE": {}, "CSI": {th: {} for th in thresholds}}

for seq_idx, sequence in enumerate(sequences):
    print(f"\nProcessing sequence {seq_idx + 1}...")

    labels = process_data(extract_labels(sequence))[:total_lead_frames]
    labels = data_preprocessing(labels)
    labels = data_postprocessing(labels)
    timestamps = [f"t+{(j + 1) * 5}" for j in range(labels.shape[0])]

    predictions_dict = {}

    for model_num, config in model_config.items():
        n_input = config["slice"].stop - config["slice"].start
        model_name = f"Model_{model_num}"

        if model_num not in model_instances:
            weight_path = f"{checkpoint_dir}/{model_num}in/weights-best.pth"
            model_instances[model_num] = load_model(n_input, n_output_frames, weight_path)

        inputs = process_data(extract_inputs(sequence, config["slice"]))
        prediction = make_predictions(
            model=model_instances[model_num],
            input_data=inputs,
            n_output_frames=n_output_frames,
            model_input_frames=n_input if n_output_frames == 6 else None
        )
        predictions_dict[model_name] = prediction

        # Metrics
        mae = calculate_MAE(labels, prediction)
        rmse = calculate_RMSE(labels, prediction)
        metrics_all["MAE"][model_name] = mae
        metrics_all["RMSE"][model_name] = rmse

        for th in thresholds:
            metrics_all["CSI"][th][model_name] = [CSI(labels[i], prediction[i], th) for i in range(labels.shape[0])]

    # Store predictions and ground truth for visualization later
    results[seq_idx] = {
        "labels": labels,
        "predictions_dict": predictions_dict,
        "timestamps": timestamps
    }

# ------------------------
# Save Metrics
# ------------------------
for metric in ["MAE", "RMSE"]:
    with open(f"{output_dir}/{metric}.txt", "w") as f:
        for model_name, values in metrics_all[metric].items():
            f.write(f"{model_name}: {values}\n")

for th in thresholds:
    with open(f"{output_dir}/CSI_threshold_{th}.txt", "w") as f:
        for model_name, values in metrics_all["CSI"][th].items():
            f.write(f"{model_name}: {values}\n")

# ------------------------
# Plot Metrics
# ------------------------
lead_time_minutes = [5 * (i + 1) for i in range(total_lead_frames)]

for metric in ["MAE", "RMSE"]:
    plt.figure(figsize=(10, 6))
    for model_num, config in model_config.items():
        model_name = f"Model_{model_num}"
        values = metrics_all[metric][model_name]
        plt.plot(lead_time_minutes, values, label=model_rename_dict[model_name], color=config["color"], marker='o')
    plt.xlabel("Lead Time (minutes)")
    plt.ylabel(metric)
    plt.title(f"{metric} over Lead Time")
    plt.grid(True)
    plt.legend()
    plt.savefig(f"{output_dir}/{metric}_plot.png", dpi=300)
    plt.close()

for th in thresholds:
    plt.figure(figsize=(10, 6))
    for model_num, config in model_config.items():
        model_name = f"Model_{model_num}"
        values = metrics_all["CSI"][th][model_name]
        plt.plot(lead_time_minutes, values, label=model_rename_dict[model_name], color=config["color"], marker='o')
    plt.xlabel("Lead Time (minutes)")
    plt.ylabel(f"CSI (>{th} mm/h)")
    plt.title(f"CSI over Lead Time (Threshold {th})")
    plt.grid(True)
    plt.legend()
    plt.savefig(f"{output_dir}/CSI_{th}_plot.png", dpi=300)
    plt.close()
    
# ------------------------ Visualization for Selected Sequences ------------------------

plot_seq_ids = [0]  # Select which sequence(s) to visualize, this will plot the predictions for all used models

for seq_id in plot_seq_ids:
    data = results[seq_id]
    labels = data["labels"]
    predictions_dict = data["predictions_dict"]
    timestamps = data["timestamps"]

    for model_name, prediction in predictions_dict.items():
        plot_animations(labels, prediction, output_dir, model_name)

    plot_predictions_frame(
        obs=labels,
        predictions_dict=predictions_dict,
        path=output_dir,
        timestamps=timestamps,
        selected_models=selected_models,
        lead_times=lead_times,
        model_rename_dict=model_rename_dict
    )
