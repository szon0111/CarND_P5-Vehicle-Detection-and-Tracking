# **Vehicle Detection Project**

[//]: # (Image References)
[image1]: https://cloud.githubusercontent.com/assets/10526591/24394334/8bb0759e-13d6-11e7-9131-b9ea57a2acd0.png "veh_HOG"
[image2]: https://cloud.githubusercontent.com/assets/10526591/24394333/8baee7e2-13d6-11e7-84ee-67af8d45ff0a.png "non-veh_HOG"
[image3]: https://cloud.githubusercontent.com/assets/10526591/24392978/3437376c-13d1-11e7-8c05-b7646cee237b.png "windows"
[image4]: https://cloud.githubusercontent.com/assets/10526591/24394284/54222e38-13d6-11e7-88ed-7cb688e6604d.png "ex1"
[image5]: https://cloud.githubusercontent.com/assets/10526591/24394288/542d0830-13d6-11e7-8843-51b8274dc19c.png "ex2"
[image6]: https://cloud.githubusercontent.com/assets/10526591/24394287/542b1746-13d6-11e7-9654-4f3cbf7a8550.png "ex3"
[image7]: https://cloud.githubusercontent.com/assets/10526591/24394285/54234c46-13d6-11e7-8d68-2c4dc1e04d96.png "heatmap"
[image8]: https://cloud.githubusercontent.com/assets/10526591/24394286/5423c20c-13d6-11e7-8abb-793da01a6b8d.png "box"
[video]: https://youtu.be/_23T4mz0IV0 "Video"

### Deliverables

#### 1. File Explanations

My project includes the following files (in image processing order):
* `train.py` to train the Linear SVM model with set parameters. The training data and parameters are saved in a pickle file for later use.
* `extract.py` to extract features using hog sub-sampling and make predictions.
* `box.py` to store windows found over a set number of frames, get heatmap, and return image with final bounding box
* `video.py` to produce the video with bounding boxes and number of detected vehicles displayed.
* `project_video_output.mp4` is the final video with smoother detection by accounting for previous frames

#### 2. How to produce output video with vehicle detection
```sh
python video.py
```

### Dataset

The dataset used for this project includes 8,792 vehicle images and 8,968 non-vehicle images in 64x64 pixels. The images were extracted from the GTI and KITTI datasets.

### Histogram of Oriented Gradients (HOG)

#### 1. HOG Feature Extraction
The code for this step is contained in the file `extract.py`. More specifically, the function `get_hog_features()` with defined HOG parameters does the work using scikit-learn's `hog()` function.


#### 2. Final HOG parameters.

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

*The above images can be reproduced by running the `extract.py` file.*

#### 3. Classifier Model Training

I trained a Linear SVM model using the parameters shown above with histogram features and spatial intensity (shown in the beginning of `train.py`). The `LinerSVM()` function in scikit-learn was used. 
The data is shuffled and divided into training and testing set before training. (lines 54 ~56).
The resulting test accuracy came out to be 98.65%.

### Sliding Window Search

#### 1. Implementations and Parameters

The function is `find_cars()` in `extract.py` and the overall pipeline can be seen in the `Box()` class in `box.py`. This function searches the HOG features for the image once, instead of repeating the feature extraction for individual windows, which greatly reduces computation time. The model outputs, parameter settings, and window settings can be seen in lines 68 ~ 89 of `box.py` and lines 10 ~ 24, lines 70 ~ 75 of `video.py`.

I limited the search are to the lower half of the image, in order to avoid detecting non-vehicle objects. I then decided to use small windows for the upper part of the search area and 2 larger sized windows for the rest of the search area. This was implemented by scaling the image in 3 different sizes(1.0, 1.5, and 2.0) The windows are also shown in 3 different colors(blue, green, red).

I decided to overlap the windows by 75%, which was the optimal value with the right balance between accuracy and computation time, by setting `cells_per_step`(line 146) in `extract.py` to 2.

Here are the windows and search areas:

![windows][image3]


#### 2. Examples on test images and Performance Optimization.

Through trial and error, I decided to use the parametes shown in the table earlier. Setting cells_per_step to 1, which gives a 87.5% overlap, gave better results but the longer computation time was not worth the small improvement. Once again, extracting hog features only once per image with the function `find_cars()` greatly improved the performance.

Here are some examples of windows on the test images:


![ex1][image4]
![ex2][image5]
---

#### Filter for False Positives

I recorded the positions of positive detections in each frame of the video.  From the 15 positive detections aggregated through the `Box()` class, I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video

**Here is the image with the window detections once again:**

![alt text][image6]

**Here is the corresponding heatmap:**

![alt text][image7]

**Here the resulting bounding boxes are drawn:**

![alt text][image8]


### Smoothing Detections

To make the detections look smoother, I stored the detection results of the previouse 15 frames before drawing the bounding boxes on the vehicles.
This way, boxes do not drastically change in size and location per frame.
The implemention can be seen in the `add_windows()` and `get_heatmap()` methods in `box.py`.


### Video Implementation

The output video is `project_video_output.mp4`.
The **[video]** is also available on Youtube


---

### Discussion

#### 1. Limitations

The major issue with the SVM approach is parameter tuning and long computation time. I felt like I am overfitting to the project video too much by tuning the parameters to get the best restults. Also, it took about 20 minutes to produce a 50 second video, which is obviously not going to work for real time detection. I am  going to explore deep learning methods such as YOLO for future work, which seems to be a great choice for real time detection.

