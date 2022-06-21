import sys
import glob
import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
from tqdm import tqdm
from skimage.morphology import skeletonize
import sys
from skimage.morphology import thin, skeletonize
from hog import hog
from process.preProcessed import pad_image
width, height = 242, 341


save_dir = "features"

data_paths = glob.glob(f"processed/*.png")

merge_size = []

for path in tqdm(data_paths):
    frame = cv2.imread(path, 0)
    
    inp_image = pad_image.pad_image(frame, width, height)
    inp_image[inp_image >= 128] = 255
    inp_image[inp_image < 128] = 0


    H = hog(inp_image)

    os.makedirs(os.path.join(save_dir), exist_ok=True)
    np.save(os.path.join(save_dir, os.path.basename(path)[:-4]) + ".npy", H)
    print(os.path.join(save_dir, os.path.basename(path)[:-4]) + ".npy")

