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
# db = sys.argv[1]

save_dir = "features"

data_paths = glob.glob(f"processed/*.png")

merge_size = []

for path in tqdm(data_paths):
    frame = cv2.imread(path, 0)
    # inp_image = enhance(frame)
    # #inp_image = preprocess(frame.copy())
    
    
    # inp_image = skeletonize(inp_image > 0)
    # inp_image = np.array(inp_image, dtype=np.uint8)
    # inp_image *= 255

    # plot_comparison(frame, inp_image, "")
    # plt.show()
    # skel = np.array(thin(inp_image > 0)).astype(np.int8)*255
    
    # inp_image = preprocess(img)

    # os.makedirs(os.path.join("processed", db), exist_ok=True)
    # cv2.imwrite(os.path.join("processed", db, os.path.basename(path)[:-4]) + ".png", inp_image)
    
    inp_image = pad_image.pad_image(frame, width, height)
    inp_image[inp_image >= 128] = 255
    inp_image[inp_image < 128] = 0


    H = hog(inp_image)

    os.makedirs(os.path.join(save_dir), exist_ok=True)
    # np.save(os.path.join(save_dir, db, os.path.basename(path)[:-4]) + ".npy", feature)
    np.save(os.path.join(save_dir, os.path.basename(path)[:-4]) + ".npy", H)
    print(os.path.join(save_dir, os.path.basename(path)[:-4]) + ".npy")

