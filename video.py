import numpy as np
import cv2
import pickle
from heatmap import get_heatmap, get_labels
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
    output_video.write_videofile(
        '{}_output.mp4'.format(input_video), audio=False)


def process(image):
    """
    Process undistortion, thresholding, warp, line detection
    Annotate curvature and vehicle position values to frame
    """
    box = Box(keep=15)
    draw_image = np.copy(image)
    hot_windows = []
    for i in range(len(ystart)):
        hot_windows.append(find_cars(image, ystart[i], ystop[i], scale[i], svc, X_scaler,
                                     orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, show_all=False))
    window_img = box.draw_boxes(draw_image, hot_windows[0],
                                color=color[0], thick=4)
    window_img = box.draw_boxes(window_img, hot_windows[1],
                                color=color[1], thick=4)
    window_img = box.draw_boxes(window_img, hot_windows[2],
                                color=color[2], thick=4)
    hot_windows = [item for sublist in hot_windows for item in sublist]

    # Add windows to Box class
    if hot_windows:
        box.add_windows(hot_windows)
    boxes = box.get_windows()
    # Get heat map
    heatmap = get_heatmap(image, boxes, threshold=15)
    # Draw final bounding boxes
    labels = get_labels(heatmap)
    draw_img = box.draw_labeled_boxes(draw_image, labels)
    # Annotate number of vehicles to video
    num_vehicles = labels[1]
    cv2.putText(draw_img, 'number of vehicles: {}'.format(num_vehicles),
                (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color=(255, 255, 255), thickness=2)

    return draw_img


if __name__ == '__main__':
    ystart = [380, 380, 380]
    ystop = [528, 592, 656]
    scale = [1.0, 1.5, 2.0]
    color = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
    video('test_video')
