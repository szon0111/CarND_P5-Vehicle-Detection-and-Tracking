# Project 5: Vehicle Detection and Tracking
[//]: # (Image References)
[video]: https://youtu.be/_23T4mz0IV0 "Video"

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)


Overview
---
Identify vehicles in a video from a front-facing camera on a car using image classifiers such as SVMs and HOG.

#### The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Apply a color transform and append binned color features, as well as histograms of color, to the HOG feature vector.
* Implement a sliding-window technique and use the trained classifier to search for vehicles in images.
* Run the pipeline on a video stream and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

Project Deliverables
---
* `train.py` to train the Linear SVM model with set parameters. The training data and parameters are saved in a pickle file for later use.
* `extract.py` to extract features using hog sub-sampling and make predictions.
* `box.py` to store windows found over a set number of frames, get heatmap, and return image with final bounding box
* `video.py` to produce the video with bounding boxes and number of detected vehicles displayed.
* `project_video_output` is the final video with smoother detection by accounting for previous frames

Results
---
View the **[video]** on Youtube