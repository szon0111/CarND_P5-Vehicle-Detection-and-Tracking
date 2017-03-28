import numpy as np
import cv2
import pickle
from scipy.ndimage.measurements import label
from moviepy.video.io.VideoFileClip import VideoFileClip

from box import Box
from extract import find_cars

# Load training data and parameters
with open('./classifier.p', mode='rb') as p:
    classifier_data = pickle.load(p)
svc = classifier_data['svc']
X_scaler = classifier_data['X_scaler']
color_space = classifier_data['color_space']
orient = classifier_data['orient']
pix_per_cell = classifier_data['pix_per_cell']
cell_per_block = classifier_data['cell_per_block']
hog_channel = classifier_data['hog_channel']
spatial_size = classifier_data['spatial_size']
hist_bins = classifier_data['hist_bins']
spatial_feat = classifier_data['spatial_feat']
hist_feat = classifier_data['hist_feat']
hog_feat = classifier_data['hog_feat']


def video(input_video):
    """
    Build video with vehicle detection boxes, number of vehicles detected
    """
    original_video = '{}.mp4'.format(input_video)
    video = VideoFileClip(original_video)
    output_video = video.fl_image(process)
    output_video.write_videofile('{}_output.mp4'.format(input_video), audio=False)


def process(image):
    """
    Extract features, search windows, add heatmap, and draw final bounding boxes
    """
    hot_windows = []
    # Append the 3 different size windows
    for i in range(len(ystart)):
        hot_windows.append(find_cars(image, ystart[i], ystop[i], scale[i], svc, X_scaler,
                                     orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, show_all=False))
    hot_windows = [item for sublist in hot_windows for item in sublist]

    print(len(hot_windows))

    # Add windows to list
    if len(hot_windows) > 0:
        box.add_windows(hot_windows)
    print(len(box.windows_list))
    # Get heatmap from aggregated windows list
    heatmap = box.get_heatmap(image)
    heatmap = box.apply_threshold(heatmap, threshold=15)
    # heatmap = np.clip(heatmap, 0, 255)
    # Draw final bounding boxes from heatmap
    labels = label(heatmap)
    draw_img = box.draw_labeled_boxes(np.copy(image), labels)
    # Annotate number of vehicles to video
    num_vehicles = labels[1]
    cv2.putText(draw_img, 'Number of vehicles: {}'.format(num_vehicles),
                (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color=(255, 255, 255), thickness=2)

    return draw_img


if __name__ == '__main__':
    ystart = [380, 380, 380]
    ystop = [528, 592, 656]
    scale = [1.0, 1.5, 2.0]
    color = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
    box = Box(keep=15)
    video('project_video')
