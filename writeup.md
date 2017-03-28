# **Vehicle Detection Project**

[//]: # (Image References)
[image1]: https://cloud.githubusercontent.com/assets/10526591/24391851/3b6d5eda-13cc-11e7-907c-b7b9e92f6ba3.png "veh_HOG"
[image2]: https://cloud.githubusercontent.com/assets/10526591/24391850/3b6c6642-13cc-11e7-8a88-fde3caacd3ce.png "non-veh_HOG"
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: https://youtu.be/_23T4mz0IV0 "Video"

### Deliverables

#### 1. File Explanations

My project includes the following files (in image processing order):
* `train.py` to train the Linear SVM model with set parameters. The training data and parameters are saved in a pickle file for later use.
* 'extract.py` to extract features using hog sub-sampling and make predictions.
* `box.py` to store windows found over a set number of frames, get heatmap, and return image with final bounding box
* `video.py` to produce the video with bounding boxes and number of detected vehicles displayed.
* `project_video_output` is the final video with smoother detection by accounting for previous frames

#### 2. How to produce output video with vehicle detection
```sh
python video.py
```

### Dataset

The dataset used for this project includes 8,792 vehicle images and 8,968 non-vehicle images in 64x64 pixels. The images were extracted from the GTI and KITTI datasets.

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how you extracted HOG features from the training images.

The code for this step is contained in the file `extract.py`. More specifically, the function `get_hog_features() with defined HOG parameters does the work using scikit-learn's `hog()` function.


#### 2. Explain how you settled on your final choice of HOG parameters.

To extract the optimal HOG features, I tried different color spaces, number of HOG orienations, HOG channels, pixels per cell, and cells per block.
After several trial-and-errors and accounting for computation time, the following parameters were chosen:

| Parameter        | Value   | 
|:-------------:|:-------------:| 
| Color Space      | YCrCb        | 
| # Orientations      | 9      |
| Channels     | 'All'      |
| Pixels per Cell      | 8        |
| Cells per Block      | 2        |

The above settings are saved in a pickle file via `train.py`.

Here are examples of HOG features on vechile and non-vehicle images (gray colorspace):

![veh][image1]
![non-veh][image2]

*The above images can be reproduced by running the `extract.py` file.

#### 3. Describe how  you trained a classifier using your selected HOG features.

I trained a Linear SVM model using the parameters shown above with histogram features and spatial intensity (shown in the beginning of `train.py`). The `LinerSVM()` function in scikit-learn was used. 
The data is shuffled and divided into training and testing set before training. (lines 54 ~56).
The resulting test accuracy came out to be 98.65%.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to set different search areas for 3 different window sizes. The function is `find_cars()` in `extract.py` and the 

![alt text][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
---

### Video Implementation

The output video is `project_video_output.mp4`.
The **[video]** is also available on Youtube


#### 2. Describe how  you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The major issue with the SVM approach is parameter tuning and long computation time. It took about 20 minutes to produce a 50 second video. I will explore deep learning methods such as YOLO for future work.

