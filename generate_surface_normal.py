import os
import shutil

import imageio.v2 as imageio
import numpy as np
import torch
import cv2
from rich.progress import track
import torch.nn.functional as F


def depth_to_surface_normals(depth: torch.Tensor, surfnorm_scalar=256) -> torch.Tensor:
    SURFNORM_KERNEL = torch.from_numpy(
        np.array(
            [
                [[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
                [[1, 0, -1], [2, 0, -2], [1, 0, -1]],
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            ]
        )
    )[:, np.newaxis, ...].to(dtype=torch.float32, device=depth.device)
    with torch.no_grad():
        surface_normals = F.conv2d(depth, surfnorm_scalar * SURFNORM_KERNEL, padding=1)
        surface_normals[:, 2, ...] = 1
        surface_normals = surface_normals / surface_normals.norm(dim=1, keepdim=True)
    return surface_normals


dataset = "/home/huaizhi_qu/Downloads/Replica_Dataset"
scenes = os.listdir(dataset)
for scene in scenes:
    if scene == "semantic_info" or scene == "readme.txt":
        continue
    sequences = os.listdir(dataset + "/" + scene)
    for sequence in sequences:
        depth_map_path = os.path.join(dataset, scene, sequence, "depth")
        print("now reading depth map from {}".format(depth_map_path))
        surface_normal_path = os.path.join(dataset, scene, sequence, "surface_normal")
        print("now saving surface normal to {}\n".format(surface_normal_path))
        img_num = len(os.listdir(depth_map_path))
        imgs = []
        for i in range(img_num):
            imgs.append(
                torch.Tensor(
                    imageio.imread(depth_map_path + "/" + f"depth_{i}.png").astype(
                        float
                    )
                )
            )
        depth_maps = torch.stack(imgs).unsqueeze(1)
        surface_normals = (
            ((depth_to_surface_normals(depth_maps).permute(0, 2, 3, 1) + 1) * (255 / 2))
            .numpy()
            .astype(np.uint8)
        )
        if not os.path.exists(surface_normal_path):
            os.makedirs(surface_normal_path)
        else:
            shutil.rmtree(surface_normal_path)
            os.makedirs(surface_normal_path)
        for i in track(range(len(surface_normals))):
            # torch.save(surface_normals[i].clone(), surface_normal_path + f"/surface_normal_{i}.pt")
            cv2.imwrite(
                surface_normal_path + f"/surface_normal_{i}.png", surface_normals[i]
            )
        del imgs
        del depth_maps
        del surface_normals
