import glob
import numpy as np
from tqdm import tqdm
from hog import hog
import cv2
import os
import compare
from process.preProcessed import pad_image

def similar(valid_path):
    data = glob.glob("features/*.npy")
    frame = cv2.imread(valid_path, 0)
    inp_image = pad_image.pad_image(frame, 242, 341)
    inp_image[inp_image >= 128] = 255
    inp_image[inp_image < 128] = 0
    anchor = hog(inp_image)
    print(anchor.shape)
    distances = []
    for path in data:
        feature = np.load(path)
        distances.append(compare.find_cosine_similarity(anchor, feature))
    idx = np.argmax(np.array(distances))
    print(data[idx]) 
    print (os.path.basename(data[idx])[:-4] + ".png")
    return os.path.basename(data[idx])[:-4] + ".png"
