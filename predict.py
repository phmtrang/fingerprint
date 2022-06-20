import glob
import numpy as np
from tqdm import tqdm
from hog import hog
import cv2
import os
from process.preProcessed import pad_image

def distance(fp_encodings, fp_to_compare):
    return np.linalg.norm(fp_encodings - fp_to_compare, axis=0)
def find_cosine_distance(source_representation, test_representation):
    
    a = np.matmul(np.transpose(source_representation), test_representation)
    b = np.sum(np.multiply(source_representation, source_representation))
    c = np.sum(np.multiply(test_representation, test_representation))
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))

def find_euclidean_distance(source_representation, test_representation):
    if type(source_representation) == list:
        source_representation = np.array(source_representation)

    if type(test_representation) == list:
        test_representation = np.array(test_representation)

    euclidean_distance = source_representation - test_representation
    euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
    euclidean_distance = np.sqrt(euclidean_distance)
    return euclidean_distance

def similar(valid_path):
    data = glob.glob("features/*.npy")
    frame = cv2.imread(valid_path, 0)
    inp_image = pad_image.pad_image(frame, 242, 341)
    inp_image[inp_image >= 128] = 255
    inp_image[inp_image < 128] = 0
    anchor = hog(inp_image)
    distances = []
    for path in data:
        feature = np.load(path)
        distances.append(find_cosine_distance(anchor, feature))
    idx = np.argmin(np.array(distances))
    print(data[idx]) 
    print (os.path.basename(data[idx])[:-4] + ".png")
    return os.path.basename(data[idx])[:-4] + ".png"
