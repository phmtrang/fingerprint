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

db = sys.argv[1]

save_dir = "features"

data_paths = glob.glob(f"fingerprints/{db}/*.tif")

merge_size = []

for path in tqdm(data_paths):
    frame = cv2.imread(path, 0)
    # inp_image = enhance(frame.copy())
    inp_image = preprocess(frame.copy())
    # inp_image_ = fingerprint_enhancer.enhance_Fingerprint(frame.copy())
    inp_image = np.array(inp_image, dtype=np.uint8)
    inp_image *= 255

    # Terminations, Bifurcations = fingerprint_feature_extractor.extract_minutiae_features(img)
    # FeaturesTerminations, FeaturesBifurcations = [], []
    # for i in Terminations:
    #     FeaturesTerminations.append([i.locX, i.locY, i.Orientation])
    # for i in Bifurcations:
    #     FeaturesBifurcations.append([i.locX, i.locY, i.Orientation])
    #break
    
    inp_image = skeletonize(inp_image > 0)
    inp_image = np.array(inp_image, dtype=np.uint8)
    inp_image *= 255

    plot_comparison(frame, inp_image, "")
    plt.show()
    # skel = np.array(thin(inp_image > 0)).astype(np.int8)*255
    
    # inp_image = preprocess(img)

    os.makedirs(os.path.join("processed", db), exist_ok=True)
    cv2.imwrite(os.path.join("processed", db, os.path.basename(path)[:-4]) + ".png", inp_image)

    inp_image = pad_image(inp_image)
    inp_image[inp_image >= 128] = 255
    inp_image[inp_image < 128] = 0

    # (H, hogImage) = feature.hog(inp_image, orientations=9, pixels_per_cell=(8, 8),
    #     cells_per_block=(2, 2), transform_sqrt=False, block_norm="L2",
    #     visualize=True)
    H = hog(inp_image)

    os.makedirs(os.path.join(save_dir, db), exist_ok=True)
    # np.save(os.path.join(save_dir, db, os.path.basename(path)[:-4]) + ".npy", feature)
    np.save(os.path.join(save_dir, db, os.path.basename(path)[:-4]) + ".npy", H)
    print(os.path.join(save_dir, db, os.path.basename(path)[:-4]) + ".npy")
    # hogImage = exposure.rescale_intensity(hogImage, out_range=(0, 255))
    # hogImage = hogImage.astype("uint8")
#     cv2.imshow("origin", frame)
#     cv2.imshow("image", inp_image)
#     # cv2.imshow("HOGimage", hogImage)
#     k = cv2.waitKey(0)
#     if k == ord('s'):
#         cv2.imwrite("image/origin.png", frame)
#         cv2.imwrite("image/extract_roi.png", inp_image)
#     if k == ord("q"):
#         break
# cv2.destroyAllWindows()
