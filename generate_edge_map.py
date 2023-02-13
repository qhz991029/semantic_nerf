import os
import shutil

import numpy as np
import cv2
from rich.progress import track


dataset = "/home/sw99/huaizhi_qu/Replica_Dataset"
scenes = os.listdir(dataset)
for scene in scenes:
    if scene == "semantic_info" or scene == "readme.txt" or scene == "meta_info":
        continue
    sequences = os.listdir(dataset + "/" + scene)
    for sequence in sequences:

        image_path = os.path.join(dataset, scene, sequence, "rgb")
        print("now reading depth map from {}".format(image_path))

        edge_map_path = os.path.join(dataset, scene, sequence, "edges")
        print("now saving surface normal to {}\n".format(edge_map_path))

        if not os.path.exists(edge_map_path):
            os.makedirs(edge_map_path)
        else:
            shutil.rmtree(edge_map_path)
            os.makedirs(edge_map_path)

        img_num = len(os.listdir(image_path))
        for i in track(range(img_num)):
            img = cv2.imread(image_path + "/" + f"rgb_{i}.png", cv2.IMREAD_GRAYSCALE)
            edge_map = cv2.Canny(img, 100, 200, L2gradient=True)
            cv2.imwrite(edge_map_path + f"/edge_{i}.png", edge_map)
