#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities for rainfall nowcasting: preprocessing, postprocessing,
model loading, and prediction helpers.

Author: Ahmed Abdelhalim
Date: July 2025
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import imageio.v3 as iio

# For cartographic plots in plot_animations()
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from nowcast_unet import NowcastUNet

# Set global device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ------------------------- #
#      Scaling Utilities     #
# ------------------------- #

def Scaler(array):
    """Apply log-scaling to rain intensities."""
    if isinstance(array, torch.Tensor):
        return torch.log10(array + 1)
    return np.log10(array + 1)

def invScaler(array):
    """Invert log-scaling."""
    if isinstance(array, torch.Tensor):
        return torch.pow(10, array) - 1
    return 10 ** array - 1

# ------------------------- #
#   Preprocessing / Post    #
# ------------------------- #

def data_preprocessing(X, device=DEVICE):
    if not isinstance(X, torch.Tensor):
        X = torch.tensor(X, dtype=torch.float32)
    return torch.log10((X / 32.0) + 1).to(device)

def data_postprocessing(X):
    if isinstance(X, torch.Tensor):
        X = torch.pow(10, X) - 1
        X = torch.clamp(X, min=0)
        return X.cpu().numpy()
    return np.where((10 ** X - 1) > 0, 10 ** X - 1, 0)

# ------------------------- #
#       Model Loading       #
# ------------------------- #

def load_model(n_input_frames, n_output_frames, checkpoint_path):
    """Load NowcastUNet model with specified input/output lengths and weights."""
    model = NowcastUNet(n_input_frames=n_input_frames, n_output_frames=n_output_frames)
    model.load_state_dict(torch.load(checkpoint_path, map_location='cpu', weights_only=True), strict=False)
    return model

# ------------------------- #
#   Prediction - General    #
# ------------------------- #
def prediction_seq2seq(model, input_data, device=DEVICE):
    model = model.to(device).eval()
    input_data = data_preprocessing(input_data[np.newaxis], device)
    with torch.no_grad():
        output = model(input_data).cpu()
    return data_postprocessing(output)

def prediction_recursive_1step(model, input_data, lead_time, device=DEVICE):
    model = model.to(device).eval()
    input_data = data_preprocessing(input_data[np.newaxis], device)
    preds = []
    with torch.no_grad():
        for _ in range(lead_time):
            pred = model(input_data).squeeze(1)
            preds.append(pred.cpu().numpy())
            input_data = torch.cat([input_data[:, 1:], pred.unsqueeze(1)], dim=1)
    return data_postprocessing(np.concatenate(preds, axis=0))

def prediction_recursive_6step(model, input_data, model_input_frames=4, step_size=6, total_steps=12, device=DEVICE):
    model = model.to(device).eval()
    input_data = data_preprocessing(input_data[np.newaxis], device)
    preds = []
    with torch.no_grad():
        for _ in range(total_steps // step_size):
            pred = model(input_data)
            preds.append(pred.cpu().numpy())
            input_data = torch.cat([input_data[:, step_size:], pred], dim=1)
    return data_postprocessing(np.concatenate(preds, axis=1).squeeze(0))

# ------------------------- #
#   Dispatcher Function     #
# ------------------------- #
def make_predictions(model, input_data, n_output_frames, model_input_frames=None):
    if n_output_frames == 12:
        return prediction_seq2seq(model, input_data)
    elif n_output_frames == 6:
        return prediction_recursive_6step(model, input_data, model_input_frames=model_input_frames, step_size=6, total_steps=12)
    elif n_output_frames == 1:
        return prediction_recursive_1step(model, input_data, lead_time=12)
    else:
        raise ValueError("Unsupported n_output_frames: {}".format(n_output_frames))

# ------------------------- #
#     Data I/O Helpers      #
# ------------------------- #

def extract_inputs(sequence, input_range):
    return sequence[input_range]

def extract_labels(sequence):
    return sequence[24:36]

def process_data(filenames):
    return np.array([np.array(iio.imread(f), dtype=np.float32) for f in filenames])

# ------------------------- #
#      Visualization        #
# ------------------------- #

def plot_animations(obs, pre, path, model_name, cmap='rainbow'):
    crs_uk = ccrs.OSGB()
    x_min, x_max, y_min, y_max = 150000, 662000, 0, 512000
    fig1, ax1 = plt.subplots(subplot_kw={'projection': crs_uk})
    fig2, ax2 = plt.subplots(subplot_kw={'projection': crs_uk})
    obs_ims, pre_ims = [], []

    for i in range(obs.shape[0]):
        t_label = f't + {i + 1}'
        line1 = ax1.annotate(t_label, xy=(0.05, 0.95), xycoords='axes fraction')
        line2 = ax2.annotate(t_label, xy=(0.05, 0.95), xycoords='axes fraction')

        obs_frame, pre_frame = np.copy(obs[i]), np.copy(pre[i])
        obs_frame[obs_frame < 0.5] = np.nan
        pre_frame[pre_frame < 0.5] = np.nan

        obs_im = ax1.imshow(obs_frame, vmin=0, vmax=10, animated=True, cmap=cmap,
                            transform=crs_uk, extent=[x_min, x_max, y_min, y_max])
        pre_im = ax2.imshow(pre_frame, vmin=0, vmax=10, animated=True, cmap=cmap,
                            transform=crs_uk, extent=[x_min, x_max, y_min, y_max])

        obs_ims.append([obs_im, line1])
        pre_ims.append([pre_im, line2])

    for ax in [ax1, ax2]:
        ax.coastlines(resolution='10m')
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        ax.set_extent([x_min, x_max, y_min, y_max], crs=crs_uk)
        ax.gridlines(draw_labels=True, x_inline=False, y_inline=False).top_labels = False

    animation.ArtistAnimation(fig1, obs_ims, interval=150).save(f"{path}/obs_{model_name}.gif")
    animation.ArtistAnimation(fig2, pre_ims, interval=150).save(f"{path}/pre_{model_name}.gif")
    plt.show()
    return path

def plot_predictions_frame(obs, predictions_dict, path, timestamps, cmap='rainbow',
                           selected_models=None, lead_times=None, model_rename_dict=None):
    if lead_times is None:
        lead_times = range(obs.shape[0])
    if selected_models is None:
        selected_models = list(predictions_dict.keys())
    if model_rename_dict is None:
        model_rename_dict = {}

    models = ['Ground Truth'] + [model_rename_dict.get(m, m) for m in selected_models]
    fig, axes = plt.subplots(len(lead_times), len(models), figsize=(4 * len(models), 3 * len(lead_times)), constrained_layout=True)

    for row, t in enumerate(lead_times):
        for col, model in enumerate(models):
            ax = axes[row, col]
            data = obs[t] if col == 0 else predictions_dict[selected_models[col - 1]][t]
            data = np.copy(data)
            data[data < 0.5] = np.nan
            im = ax.imshow(data, cmap=cmap, vmin=0, vmax=10)
            if row == 0:
                ax.set_title(model, fontsize=14)
            if col == 0:
                ax.set_ylabel(timestamps[t], fontsize=12)
            ax.set_xticks([]), ax.set_yticks([])

    cbar = fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.02, pad=0.02)
    cbar.set_label('Rainfall Intensity (mm/h)', size=12)
    plt.savefig(f"{path}/predictions_grid.png", dpi=300)
    plt.show()