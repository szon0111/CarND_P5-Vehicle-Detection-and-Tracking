import numpy as np
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
import cv2
import collections
from scipy.ndimage.measurements import label

from extract import find_cars


class Box():
    """
    Store windows found over a set number of frames
    Get heatmap from windows, based on threshold value
    """
    def __init__(self, keep=15):
        self.windows_list = collections.deque(maxlen=keep)

    def add_windows(self, new_windows):
        self.windows_list.append(new_windows)

    def get_heatmap(self, img):
        heat = np.zeros_like(img[:, :, 0]).astype(np.float)
        for windows in self.windows_list:
            for window in windows:
                heat[window[0][1]:window[1][1], window[0][0]:window[1][0]] += 1

        return heat

    def apply_threshold(self, heatmap, threshold):
        # Zero out pixels below the threshold
        heatmap[heatmap <= threshold] = 0

        # Return thresholded map
        return heatmap

    def draw_boxes(self, img, bboxes, color=(0, 0, 255), thick=6):
        # Make a copy of the image
        imcopy = np.copy(img)
        # Iterate through the bounding boxes
        for bbox in bboxes:
            # Draw a rectangle given bbox coordinates
            cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
        # Return the image copy with boxes drawn
        return imcopy

    def draw_labeled_boxes(self, img, labels):
        # Iterate through all detected cars
        for car_number in range(1, labels[1] + 1):
            # Find pixels with each car_number label value
            nonzero = (labels[0] == car_number).nonzero()
            # Identify x and y values of those pixels
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            # Define a bounding box based on min/max x and y
            box = ((np.min(nonzerox), np.min(nonzeroy)),
                   (np.max(nonzerox), np.max(nonzeroy)))
            # Draw the box on the image
            cv2.rectangle(img, box[0], box[1], (0, 0, 255), 6)
        # Return the image
        return img


if __name__ == '__main__':
    # load training data and parameters
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

    # Set search area, window size, and window color
    ystart = [380, 380, 380]
    ystop = [528, 592, 656]
    scale = [1.0, 1.5, 2.0]
    color = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
    # Set keep=1 to show heatmap for each image
    box = Box(keep=1)

    # Display predictions on all test_images
    test_img_list = glob.glob("./test_images/*.jpg")
    for img in test_img_list:
        image = mpimg.imread(img)
        hot_windows = []
        # Append the 3 different size windows
        for i in range(len(ystart)):
            hot_windows.append(find_cars(image, ystart[i], ystop[i], scale[i], svc, X_scaler,
                                         orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, show_all=False))
        window_img = box.draw_boxes(np.copy(image), hot_windows[0],
                                    color=color[0], thick=4)
        window_img = box.draw_boxes(window_img, hot_windows[1],
                                    color=color[1], thick=4)
        window_img = box.draw_boxes(window_img, hot_windows[2],
                                    color=color[2], thick=4)
        print(len(hot_windows))
        hot_windows = [item for sublist in hot_windows for item in sublist]
        print(len(hot_windows))

        # Show image with windows
        plt.imshow(window_img)
        plt.show()

        # Get heat map
        box.add_windows(hot_windows)
        heatmap = box.get_heatmap(image)
        heatmap = box.apply_threshold(heatmap, threshold=10)
        heatmap = np.clip(heatmap, 0, 255)
        # Show heatmap
        plt.imshow(heatmap, cmap='hot')
        plt.show()

        # Show image with final bounding boxes
        labels = label(heatmap)
        draw_img = box.draw_labeled_boxes(np.copy(image), labels)
        print(labels[1], 'cars found')
        plt.imshow(draw_img)
        plt.show()
