import numpy as np
from scipy.ndimage.measurements import label


def add_heat(heatmap, hot_windows):
    for box in hot_windows:
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    return heatmap


def apply_threshold(heatmap, threshold):
    heatmap[heatmap <= threshold] = 0
    return heatmap


def get_heatmap(image, hot_windows, threshold=10):
    heat = np.zeros_like(image[:, :, 0]).astype(np.float)
    heat = add_heat(heat, hot_windows)
    heat = apply_threshold(heat, threshold)
    heatmap = np.clip(heat, 0, 255)
    return heatmap


def get_labels(heatmap):
    labels = label(heatmap)
    return labels