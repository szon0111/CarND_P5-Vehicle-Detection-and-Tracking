import numpy as np
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
import cv2
from heatmap import get_heatmap, get_labels
from extract import find_cars


class Box:

    def __init__(self, keep=15):
        self.keep = keep
        self.windows = []

    def add_windows(self, new_windows):
        self.windows.append(new_windows)
        queue = len(self.windows)
        if queue >= self.keep:
            del self.windows[-1]

    def get_windows(self):
        all_windows = []
        for window in self.windows:
            all_windows += window

        return all_windows

    def draw_boxes(self, img, bboxes, color=(0, 0, 255), thick=6):
        imcopy = np.copy(img)
        for bbox in bboxes:
            cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)

        return imcopy

    def draw_labeled_boxes(self, img, labels):
        for car_number in range(1, labels[1] + 1):
            nonzero = (labels[0] == car_number).nonzero()
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            box = (
                (
                    np.min(nonzerox), np.min(nonzeroy)),
                (
                    np.max(nonzerox), np.max(nonzeroy)))
            cv2.rectangle(img, box[0], box[1], (0, 0, 255), 6)

        return img


if __name__ == '__main__':
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
    ystart = [380, 380, 380]
    ystop = [528, 592, 656]
    scale = [1.0, 1.5, 2.0]
    color = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
    box = Box()
    test_img_list = glob.glob('./test_images/*.jpg')
    for img in test_img_list:
        image = mpimg.imread(img)
        draw_image = np.copy(image)
        hot_windows = []
        for i in range(len(ystart)):
            hot_windows.append(find_cars(image, ystart[i], ystop[i], scale[i], svc, X_scaler,
                                         orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, show_all=False))

        window_img = box.draw_boxes(draw_image, hot_windows[0], color=color[0], thick=4)
        window_img = box.draw_boxes(window_img, hot_windows[1], color=color[1], thick=4)
        window_img = box.draw_boxes(window_img, hot_windows[2], color=color[2], thick=4)
        hot_windows = [item for sublist in hot_windows for item in sublist]
        
        plt.imshow(window_img)
        plt.show()

        heatmap = get_heatmap(image, hot_windows, threshold=15)
        plt.imshow(heatmap, cmap='hot')
        plt.show()

        labels = get_labels(heatmap)
        draw_img = box.draw_labeled_boxes(np.copy(image), labels)
        print(labels[1], 'cars found')
        plt.imshow(draw_img)
        plt.show()
