import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
import pickle
import glob
import time
import extract

# Set parameters
color_space = 'YUV'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb, GRAY
orient = 9  # HOG orientations
pix_per_cell = 8  # HOG pixels per cell
cell_per_block = 2  # HOG cells per block
hog_channel = 'ALL'  # Can be 0, 1, 2, or "ALL"
spatial_size = (16, 16)  # Spatial binning dimensions
hist_bins = 32  # Number of histogram bins
spatial_feat = True  # Spatial features on or off
hist_feat = True  # Histogram features on or off
hog_feat = True  # HOG features on or off

# Save images to list
print("Reading in images...")
veh_img_list = glob.glob("./dataset/vehicles/**/*.png")
non_veh_img_list = glob.glob("./dataset/non-vehicles/**/*.png")
print("Numer of vehicle images: {}".format(len(veh_img_list)))
print("Number of non-vehicle images: {}".format(len(non_veh_img_list)))

# Extract features from list of images
print("Extracting vechicle features...")
veh_features = extract.extract_features(veh_img_list, color_space=color_space,
                                        spatial_size=spatial_size, hist_bins=hist_bins,
                                        orient=orient, pix_per_cell=pix_per_cell,
                                        cell_per_block=cell_per_block,
                                        hog_channel=hog_channel, spatial_feat=spatial_feat,
                                        hist_feat=hist_feat, hog_feat=hog_feat)
print("Extracting non-vehicle features...")
non_veh_features = extract.extract_features(non_veh_img_list, color_space=color_space,
                                            spatial_size=spatial_size, hist_bins=hist_bins,
                                            orient=orient, pix_per_cell=pix_per_cell,
                                            cell_per_block=cell_per_block,
                                            hog_channel=hog_channel, spatial_feat=spatial_feat,
                                            hist_feat=hist_feat, hog_feat=hog_feat)

# Define a labels vector based on features lists
y = np.hstack((np.ones(len(veh_features)), np.zeros(len(non_veh_features))))
# Create an array stack of feature vectors
X = np.vstack((veh_features, non_veh_features)).astype(np.float64)
# Fit a per-column scaler
X_scaler = StandardScaler().fit(X)
# Apply the scaler to X
scaled_X = X_scaler.transform(X)
# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(
    scaled_X, y, test_size=0.2, random_state=rand_state)
# Use a linear SVC (support vector classifier)
svc = LinearSVC()
# Train the SVC
t = time.time()
print("Training SVC...")
svc.fit(X_train, y_train)
# Print time to train SVC in seconds
print(round(time.time() - t, 2), "seconds to train SVC")
# Print the accuracy of SVC
print("Accuracy of SVC: {:.4f}".format(svc.score(X_test, y_test)))

# Save the  training data and parameters for later use
print("Saving classifier data as pickle file...")
classifier_pickle = {'svc': svc,
                     'X_scaler': X_scaler,
                     'color_space': color_space,
                     'orient': orient,
                     'spatial_size': spatial_size,
                     'hist_bins': hist_bins,
                     'pix_per_cell': pix_per_cell,
                     'cell_per_block': cell_per_block,
                     'hog_channel': hog_channel,
                     'spatial_feat': spatial_feat,
                     'hist_feat': hist_feat,
                     'hog_feat': hog_feat
                     }

with open('./classifier.p', mode='wb') as p:
    pickle.dump(classifier_pickle, p, pickle.HIGHEST_PROTOCOL)
print("Classifier data saved")
