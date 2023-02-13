import os
import shutil

import numpy as np
import torch
import cv2
from rich.progress import track
import torch.nn.functional as F


dataset = "/home/sw99/huaizhi_qu/Replica_Dataset"
scenes = os.listdir(dataset)
for scene in scenes:
    if scene == "semantic_info" or scene == "readme.txt" or scene == "meta_info":
        continue
    sequences = os.listdir(dataset + "/" + scene)
    for sequence in sequences:

        image_path = os.path.join(dataset, scene, sequence, "rgb")
        print("now reading depth map from {}".format(image_path))

        keypoint_map_path = os.path.join(dataset, scene, sequence, "keypoint")
        print("now saving surface normal to {}\n".format(keypoint_map_path))

        if not os.path.exists(keypoint_map_path):
            os.makedirs(keypoint_map_path)
        else:
            shutil.rmtree(keypoint_map_path)
            os.makedirs(keypoint_map_path)

        sift = cv2.SIFT_create()
        # surf = cv2.xfeatures2d.SURF_create(400)

        img_num = len(os.listdir(image_path))
        x = []
        y = []
        for i in track(range(img_num)):
            img = cv2.imread(image_path + "/" + f"rgb_{i}.png", cv2.IMREAD_GRAYSCALE)
            keypoints = sift.detect(img, None)
            # keypoints = surf.detect(img, None)
            keypoint_map = np.zeros([480, 640]).astype(np.uint8)
            for keypoint in keypoints:
                # keypoint_map[keypoint.pt] = 255
                y = int(keypoint.pt[0])
                x = int(keypoint.pt[1])
                keypoint_map[x, y] = 255
            cv2.imwrite(keypoint_map_path + f"/keypoint_{i}.png", keypoint_map)
