import argparse
import os
import send2trash
import cv2
import random
#import albumentations
from tqdm import tqdm

# Build argument parser
parser = argparse.ArgumentParser(description='Image Preprocessing')

parser.add_argument('--num-images', type=int, default=1000,
    help='number of images to preprocess for each category')
args = vars(parser.parse_args())

print(f"Preprocessing {args['num_images']} from each category...")

HOME = r'C:\Users\Shreya Basu\Workspace\ASL-Translator\Project\img_data'
DATASET = r'asl_alphabet_train\asl_alphabet_train'
img_paths = os.listdir(os.path.join(HOME, DATASET))
img_paths.sort()

# Get num_images from each class folder
for i, img_path in tqdm(enumerate(img_paths), total=len(img_paths)):
    # Locate all images under a specific class directory
    images = os.listdir(os.path.join(HOME, DATASET, img_path))
    # Create a class folder withinthe newly created 'preprocessed_images' folder
    dest_path = os.path.join(HOME, 'preprocessed_images', img_path)
    if not os.path.isdir(dest_path):
        os.makedirs(dest_path)
    # Randomly select from images
    for j in range(args['num_images']):
        rand_idx = random.randint(0,2999)
        image = cv2.imread(os.path.join(HOME, DATASET, img_path, images[rand_idx]))
        image = cv2.resize(image, (224, 224))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if not cv2.imwrite(os.path.join(dest_path, f'{img_path}{j+1000}.jpg'), image):
            raise Exception("Could not write image")


print('DONE')