import pandas as pd
import numpy as np
import os
import glob
import joblib

from sklearn.preprocessing import LabelBinarizer
from tqdm import tqdm
from imutils import paths

# Get all image paths
DATASET_PATH = r'..\img_data\preprocessed_images'
img_paths = []
categories = os.listdir(DATASET_PATH)
categories.sort()
for category in categories:
    img_paths.extend(glob.glob(os.path.join(DATASET_PATH, category, '*')))

# Create empty dataframe and labels array
data = pd.DataFrame()
labels = np.array([])

print('Creating one hot encoded labels array for representing each input image...')
for i, img_path in tqdm(enumerate(img_paths), total = len(img_paths)):
    path_arr = img_path.split(os.path.sep)
    data.loc[i, 'image_path'] = os.path.join('preprocessed_images', path_arr[-2], path_arr[-1])
    labels = np.append(labels, path_arr[-2])

lb = LabelBinarizer()
labels = lb.fit_transform(labels)

print(f"The first one hot encoded labels: {labels[0]}")
print(f"Mapping the first one hot encoded label to its category: {lb.classes_[0]}")
print(f"Total instances: {len(labels)}")

print('Adding targets column to dataframe...')
for i, label in tqdm(enumerate(labels), total = len(labels)):
    idx = int(np.argmax(label))
    data.loc[i, 'target'] = idx

# Shuffle the dataset
data = data.sample(frac=1).reset_index(drop=True)

# Save as CSV
data.to_csv(r'..\img_data\data.csv')

# Pick hot-coded labels
print('Saving the binarized labels as pickled files')
joblib.dump(lb, r'..\Project\outputs\lb.pkl')

print(data.head(10))
