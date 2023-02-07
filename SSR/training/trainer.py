import json
import logging
import math
import os
import time
from collections import defaultdict

import imageio
import numpy as np
import rich
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from imgviz import label_colormap, depth2rgb
from torchmetrics.classification import MulticlassJaccardIndex
import pandas as pd
from timm import optim

from tqdm import tqdm
from SSR.models.model_utils import raw2outputs
from SSR.models.model_utils import run_network
from SSR.models.rays import sampling_index, sample_pdf, create_rays
from SSR.models.semantic_nerf import get_embedder, Semantic_NeRF
from SSR.training.training_utils import (
    batchify_rays,
    calculate_segmentation_metrics,
    calculate_depth_metrics,
)
from SSR.utils import image_utils
from SSR.visualisation.tensorboard_vis import TFVisualizer


def select_gpus(gpus):
    """
    takes in a string containing a comma-separated list
    of gpus to make visible to pytorch, e.g. '0,1,3'
    """
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    if gpus != "":
        logging.info("Using gpu's: {}".format(gpus))
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    else:
        logging.info("Using all available gpus")


class SSRTrainer(object):
    def __init__(self, config):
        super(SSRTrainer, self).__init__()
        self.config = config
        self.set_params()

        self.training = True  # training mode by default
        self.img_loss = []
        self.sem_loss = []
        self.depth_loss = []
        self.norm_loss = []
        self.psnr_coarse = []
        self.psnr_fine = []
        # create tfb Summary writers and folders
        tf_log_dir = os.path.join(config["experiment"]["save_dir"], "tfb_logs")
        if not os.path.exists(tf_log_dir):
            os.makedirs(tf_log_dir)
        self.tfb_viz = TFVisualizer(
            tf_log_dir, config["logging"]["step_log_tfb"], config
        )

    def save_config(self):
        # save config to save_dir for the convience of checking config later
        with open(
            os.path.join(self.config["experiment"]["save_dir"], "exp_config.yaml"), "w"
        ) as outfile:
            yaml.dump(self.config, outfile, default_flow_style=False)

    def set_params_replica(self):
        self.H = self.config["experiment"]["height"]
        self.W = self.config["experiment"]["width"]

        self.n_pix = self.H * self.W
        self.aspect_ratio = self.W / self.H

        self.hfov = 90
        # the pin-hole camera has the same value for fx and fy
        self.fx = self.W / 2.0 / math.tan(math.radians(self.hfov / 2.0))
        # self.fy = self.H / 2.0 / math.tan(math.radians(self.yhov / 2.0))
        self.fy = self.fx
        self.cx = (self.W - 1.0) / 2.0
        self.cy = (self.H - 1.0) / 2.0
        self.near, self.far = self.config["render"]["depth_range"]
        self.c2w_staticcam = None

        # use scaled size for test and visualisation purpose
        self.test_viz_factor = int(self.config["render"]["test_viz_factor"])
        self.H_scaled = self.H // self.test_viz_factor
        self.W_scaled = self.W // self.test_viz_factor
        self.fx_scaled = self.W_scaled / 2.0 / math.tan(math.radians(self.hfov / 2.0))
        # self.fy_scaled = self.H_scaled / 2.0 / math.tan(math.radians(self.yhov / 2.0))
        self.fy_scaled = self.fx_scaled
        self.cx_scaled = (self.W_scaled - 1.0) / 2.0
        self.cy_scaled = (self.H_scaled - 1.0) / 2.0

        self.save_config()

    def set_params_scannet(self, data):
        self.H = self.config["experiment"]["height"]
        self.W = self.config["experiment"]["width"]
        self.n_pix = self.H * self.W
        self.aspect_ratio = self.W / self.H

        K = data.intrinsics
        self.fx = K[0, 0]
        self.fy = K[1, 1]
        self.cx = K[0, -1]
        self.cy = K[1, -1]
        self.near, self.far = self.config["render"]["depth_range"]
        self.c2w_staticcam = None

        # use scaled size for test and visualisation purpose
        self.test_viz_factor = int(self.config["render"]["test_viz_factor"])
        self.H_scaled = self.config["experiment"]["height"] // self.test_viz_factor
        self.W_scaled = self.config["experiment"]["width"] // self.test_viz_factor
        self.fx_scaled = self.fx / self.test_viz_factor
        self.fy_scaled = self.fy / self.test_viz_factor
        self.cx_scaled = (self.W_scaled - 0.5) / 2.0
        self.cy_scaled = (self.H_scaled - 0.5) / 2.0

        self.save_config()

    def set_params(self):
        self.enable_multitask = self.config["experiment"]["enable_multitask"]
        self.enable_semantic = self.config["experiment"]["enable_semantic"]
        self.enable_surface_normal = self.config["experiment"]["enable_surface_normal"]
        self.enable_depth = self.config["experiment"]["enable_depth"]
        # render options
        self.n_rays = (
            eval(self.config["render"]["N_rays"])
            if isinstance(self.config["render"]["N_rays"], str)
            else self.config["render"]["N_rays"]
        )

        self.N_samples = self.config["render"]["N_samples"]
        self.netchunk = (
            eval(self.config["model"]["netchunk"])
            if isinstance(self.config["model"]["netchunk"], str)
            else self.config["model"]["netchunk"]
        )

        self.chunk = (
            eval(self.config["model"]["chunk"])
            if isinstance(self.config["model"]["chunk"], str)
            else self.config["model"]["chunk"]
        )

        self.use_viewdir = self.config["render"]["use_viewdirs"]

        self.convention = self.config["experiment"]["convention"]

        self.endpoint_feat = (
            self.config["experiment"]["endpoint_feat"]
            if "endpoint_feat" in self.config["experiment"].keys()
            else False
        )

        self.N_importance = self.config["render"]["N_importance"]
        self.raw_noise_std = self.config["render"]["raw_noise_std"]
        self.white_bkgd = self.config["render"]["white_bkgd"]
        self.perturb = self.config["render"]["perturb"]

        self.no_batching = self.config["render"]["no_batching"]

        self.lrate = float(self.config["train"]["lrate"])
        self.lrate_decay = float(self.config["train"]["lrate_decay"])

        # logging
        self.save_dir = self.config["experiment"]["save_dir"]

    def prepare_data_replica(self, data, gpu=True):
        # input data is a dataloader
        self.ignore_label = -1

        # shift numpy data to torch
        train_samples = data.train_samples
        test_samples = data.test_samples

        self.train_ids = data.train_ids
        self.test_ids = data.test_ids
        self.mask_ids = data.mask_ids

        self.num_train = data.train_num
        self.num_test = data.test_num

        # preprocess semantic info
        self.semantic_classes = torch.from_numpy(data.semantic_classes)
        self.num_semantic_class = self.semantic_classes.shape[
            0
        ]  # number of semantic classes, including void class=0
        self.num_valid_semantic_class = (
            self.num_semantic_class - 1
        )  # exclude void class
        assert self.num_semantic_class == data.num_semantic_class

        json_class_mapping = os.path.join(
            self.config["experiment"]["scene_file"], "info_semantic.json"
        )
        with open(json_class_mapping, "r") as f:
            annotations = json.load(f)
        instance_id_to_semantic_label_id = np.array(annotations["id_to_label"])
        total_num_classes = len(annotations["classes"])
        assert total_num_classes == 101  # excluding void, we have 102 classes
        # assert self.num_valid_semantic_class == np.sum(np.unique(instance_id_to_semantic_label_id) >=0 )

        colour_map_np = label_colormap(total_num_classes)[
            data.semantic_classes
        ]  # select the existing class from total colour map
        self.colour_map = torch.from_numpy(colour_map_np)
        self.valid_colour_map = torch.from_numpy(
            colour_map_np[1:, :]
        )  # exclude the first colour map to colourise rendered segmentation without void index

        # plot semantic label legend
        # class_name_string = ["voild"] + [x["name"] for x in annotations["classes"] if x["id"] in np.unique(data.semantic)]
        class_name_string = ["void"] + [x["name"] for x in annotations["classes"]]
        legend_img_arr = image_utils.plot_semantic_legend(
            data.semantic_classes,
            class_name_string,
            colormap=label_colormap(total_num_classes + 1),
            save_path=self.save_dir,
        )
        # total_num_classes +1 to include void class

        # remap different semantic classes to continuous integers from 0 to num_class-1
        self.semantic_classes_remap = torch.from_numpy(
            np.arange(self.num_semantic_class)
        )

        #####training data#####
        # rgb
        self.train_image = torch.from_numpy(train_samples["image"]).to(torch.float)
        self.train_image_scaled = F.interpolate(
            self.train_image.permute(
                0,
                3,
                1,
                2,
            ),  # [N, C, H, W]
            scale_factor=1 / self.config["render"]["test_viz_factor"],
            mode="bilinear",
        ).permute(
            0, 2, 3, 1
        )  # [N, H, W, C]
        # depth
        self.train_depth = torch.from_numpy(train_samples["depth"]).to(torch.float)
        self.viz_train_depth = np.stack(
            [
                depth2rgb(dep, min_value=self.near, max_value=self.far)
                for dep in train_samples["depth"]
            ],  # a list of depth maps
            axis=0,
        )  # [num_test, H, W, 3]
        # process the depth for evaluation purpose
        self.train_depth_scaled = (
            F.interpolate(
                torch.unsqueeze(self.train_depth, dim=1).float(),
                scale_factor=1 / self.config["render"]["test_viz_factor"],
                mode="bilinear",
            )
            .squeeze(1)
            .cpu()
            .numpy()
        )
        # surface normal
        self.train_surface_normal = train_samples["surface_normal"]
        # semantic
        self.train_semantic = torch.from_numpy(train_samples["semantic_remap"])
        self.viz_train_semantic = np.stack(
            [colour_map_np[sem] for sem in self.train_semantic], axis=0
        )  # [num_test, H, W, 3]

        self.train_semantic_clean = torch.from_numpy(
            train_samples["semantic_remap_clean"]
        )
        self.viz_train_semantic_clean = np.stack(
            [colour_map_np[sem] for sem in self.train_semantic_clean], axis=0
        )  # [num_test, H, W, 3]

        # process the clean label for evaluation purpose
        self.train_semantic_clean_scaled = F.interpolate(
            torch.unsqueeze(self.train_semantic_clean, dim=1).float(),
            scale_factor=1 / self.config["render"]["test_viz_factor"],
            mode="nearest",
        ).squeeze(1)
        self.train_semantic_clean_scaled = (
            self.train_semantic_clean_scaled.cpu().numpy() - 1
        )
        # pose
        self.train_Ts = torch.from_numpy(train_samples["T_wc"]).float()

        #####test data#####
        # rgb
        self.test_image = torch.from_numpy(test_samples["image"]).to(
            torch.float
        )  # [num_test, H, W, 3]
        # scale the test image for evaluation purpose
        self.test_image_scaled = F.interpolate(
            self.test_image.permute(
                0,
                3,
                1,
                2,
            ),
            scale_factor=1 / self.config["render"]["test_viz_factor"],
            mode="bilinear",
        ).permute(0, 2, 3, 1)

        # depth
        self.test_depth = torch.from_numpy(test_samples["depth"]).to(
            torch.float
        )  # [num_test, H, W]
        self.viz_test_depth = np.stack(
            [
                depth2rgb(dep, min_value=self.near, max_value=self.far)
                for dep in test_samples["depth"]
            ],
            axis=0,
        )  # [num_test, H, W, 3]
        # process the depth for evaluation purpose
        self.test_depth_scaled = (
            F.interpolate(
                torch.unsqueeze(self.test_depth, dim=1).float(),
                scale_factor=1 / self.config["render"]["test_viz_factor"],
                mode="bilinear",
            )
            .squeeze(1)
            .cpu()
            .numpy()
        )
        # surface_normal
        self.test_surface_normal = test_samples["surface_normal"]
        # semantic
        self.test_semantic = torch.from_numpy(
            test_samples["semantic_remap"]
        )  # [num_test, H, W]

        self.viz_test_semantic = np.stack(
            [colour_map_np[sem] for sem in self.test_semantic], axis=0
        )  # [num_test, H, W, 3]

        # we only add noise to training images, therefore test images are kept intact. No need for test_remap_clean
        # process the clean label for evaluation purpose
        self.test_semantic_scaled = F.interpolate(
            torch.unsqueeze(self.test_semantic, dim=1).float(),
            scale_factor=1 / self.config["render"]["test_viz_factor"],
            mode="nearest",
        ).squeeze(1)
        self.test_semantic_scaled = (
            self.test_semantic_scaled.cpu().numpy() - 1
        )  # shift void class from value 0 to -1, to match self.ignore_label
        # pose
        self.test_Ts = torch.from_numpy(
            test_samples["T_wc"]
        ).float()  # [num_test, 4, 4]

        if gpu is True:
            self.train_image = self.train_image.cuda()
            self.train_image_scaled = self.train_image_scaled.cuda()
            self.train_depth = self.train_depth.cuda()
            self.train_surface_normal = self.train_surface_normal.cuda()
            self.train_semantic = self.train_semantic.cuda()

            self.test_image = self.test_image.cuda()
            self.test_image_scaled = self.test_image_scaled.cuda()
            self.test_depth = self.test_depth.cuda()
            self.test_surface_normal = self.test_surface_normal.cuda()
            self.test_semantic = self.test_semantic.cuda()
            self.colour_map = self.colour_map.cuda()
            self.valid_colour_map = self.valid_colour_map.cuda()

        # set the data sampling paras which need the number of training images
        if (
            self.no_batching is False
        ):  # False means we need to sample from rays of all pixels of all images instead of rays from one random image
            self.i_batch = 0
            self.rand_idx = torch.randperm(self.num_train * self.H * self.W)

        # add datasets to tfboard for comparison to rendered images
        self.tfb_viz.tb_writer.add_image(
            "Train/legend",
            np.expand_dims(legend_img_arr, axis=0),
            0,
            dataformats="NHWC",
        )
        self.tfb_viz.tb_writer.add_image(
            "Train/rgb_GT", train_samples["image"], 0, dataformats="NHWC"
        )
        self.tfb_viz.tb_writer.add_image(
            "Train/depth_GT", self.viz_train_depth, 0, dataformats="NHWC"
        )
        self.tfb_viz.tb_writer.add_image(
            "Train/vis_sem_label_GT", self.viz_train_semantic, 0, dataformats="NHWC"
        )
        self.tfb_viz.tb_writer.add_image(
            "Train/vis_sem_label_GT_clean",
            self.viz_train_semantic_clean,
            0,
            dataformats="NHWC",
        )

        self.tfb_viz.tb_writer.add_image(
            "Test/legend", np.expand_dims(legend_img_arr, axis=0), 0, dataformats="NHWC"
        )
        self.tfb_viz.tb_writer.add_image(
            "Test/rgb_GT", test_samples["image"], 0, dataformats="NHWC"
        )
        self.tfb_viz.tb_writer.add_image(
            "Test/depth_GT", self.viz_test_depth, 0, dataformats="NHWC"
        )
        self.tfb_viz.tb_writer.add_image(
            "Test/vis_sem_label_GT", self.viz_test_semantic, 0, dataformats="NHWC"
        )

    def prepare_data_replica_nyu_cnn(self, data, gpu=True):
        self.ignore_label = -1  # default value in nn.CrossEntropy

        # shift numpy data to torch
        train_samples = data.train_samples
        test_samples = data.test_samples

        self.train_ids = data.train_ids
        self.test_ids = data.test_ids
        self.mask_ids = data.mask_ids

        self.num_train = data.train_num
        self.num_test = data.test_num

        self.nyu_mode = data.nyu_mode
        # preprocess semantic info
        self.semantic_classes = torch.from_numpy(data.semantic_classes)
        self.num_semantic_class = self.semantic_classes.shape[
            0
        ]  # predicted labels from off-the-shelf CNN results
        self.num_valid_semantic_class = self.num_semantic_class - 1  # remove voud class

        if self.nyu_mode == "nyu13":
            self.num_valid_semantic_class == 13
            colour_map_np = image_utils.nyu13_colour_code
            assert colour_map_np.shape[0] == 14
            class_name_string = [
                "void",
                "bed",
                "books",
                "ceiling",
                "chair",
                "floor",
                "furniture",
                "objects",
                "painting/picture",
                "sofa",
                "table",
                "TV",
                "wall",
                "window",
            ]
        elif self.nyu_mode == "nyu34":
            self.num_valid_semantic_class == 34
            colour_map_np = image_utils.nyu34_colour_code
            assert colour_map_np.shape[0] == 35
            class_name_string = [
                "void",
                "wall",
                "floor",
                "cabinet",
                "bed",
                "chair",
                "sofa",
                "table",
                "door",
                "window",
                "picture",
                "counter",
                "blinds",
                "desk",
                "shelves",
                "curtain",
                "pillow",
                "floor",
                "clothes",
                "ceiling",
                "books",
                "fridge",
                "tv",
                "paper",
                "towel",
                "box",
                "night stand",
                "toilet",
                "sink",
                "lamp",
                "bath tub",
                "bag",
                "other struct",
                "other furntr",
                "other prop",
            ]  # 1 void class + 34 valid class
        else:
            assert False

        """
         complete NYU-40 classes ["wall", "floor", "cabinet", "bed", "chair",
        "sofa", "table", "door", "window", "book", 
        "picture", "counter", "blinds", "desk", "shelves",
        "curtain", "dresser", "pillow", "mirror", "floor",
        "clothes", "ceiling", "books", "fridge", "tv",
        "paper", "towel", "shower curtain", "box", "white board",
        "person", "night stand", "toilet", "sink", "lamp",
        "bath tub", "bag", "other struct", "other furntr", "other prop"]

        Following classes in NYU-40 are missing during conversion of Replica to NYU-40:
        10:bookshelves
        17:dresser
        19 mirror
        28:shower curtain
        30:whiteboard
        31:person
        """

        self.colour_map = torch.from_numpy(colour_map_np)  #
        self.valid_colour_map = torch.from_numpy(
            colour_map_np[1:, :]
        )  # used in func render_path to visualise rendered segmentation without void label

        legend_img_arr = image_utils.plot_semantic_legend(
            np.unique(data.semantic_classes),
            class_name_string,
            colormap=colour_map_np,
            save_path=self.save_dir,
        )

        # remap different semantic classes to continuous integers from 0 to num_class-1
        self.semantic_classes_remap = torch.from_numpy(
            np.arange(self.num_semantic_class)
        )

        #####training data#####
        # rgb
        self.train_image = torch.from_numpy(train_samples["image"])
        self.train_image_scaled = F.interpolate(
            self.train_image.permute(
                0,
                3,
                1,
                2,
            ),
            scale_factor=1 / self.config["render"]["test_viz_factor"],
            mode="bilinear",
        ).permute(0, 2, 3, 1)
        # depth
        self.train_depth = torch.from_numpy(train_samples["depth"])
        self.viz_train_depth = np.stack(
            [
                depth2rgb(dep, min_value=self.near, max_value=self.far)
                for dep in train_samples["depth"]
            ],
            axis=0,
        )  # [num_test, H, W, 3]

        # process the depth for evaluation purpose
        self.train_depth_scaled = (
            F.interpolate(
                torch.unsqueeze(self.train_depth, dim=1).float(),
                scale_factor=1 / self.config["render"]["test_viz_factor"],
                mode="bilinear",
            )
            .squeeze(1)
            .cpu()
            .numpy()
        )

        # semantic
        self.train_semantic = torch.from_numpy(train_samples["cnn_semantic"])
        self.viz_train_semantic = np.stack(
            [colour_map_np[sem] for sem in self.train_semantic], axis=0
        )

        # network predictions act as training ground-truth
        self.train_semantic_clean = torch.from_numpy(
            train_samples["cnn_semantic_clean"]
        )
        self.viz_train_semantic_clean = np.stack(
            [colour_map_np[sem] for sem in self.train_semantic_clean], axis=0
        )  # [num_test, H, W, 3]

        # scale the cnn label for evaluation purpose
        self.train_semantic_clean_scaled = F.interpolate(
            torch.unsqueeze(self.train_semantic_clean, dim=1).float(),
            scale_factor=1 / self.config["render"]["test_viz_factor"],
            mode="nearest",
        ).squeeze(1)
        self.train_semantic_clean_scaled = (
            self.train_semantic_clean_scaled.cpu().numpy() - 1
        )  # shift void class to -1

        # GT label from Replica ground-truth
        self.train_semantic_gt = torch.from_numpy(train_samples["gt_semantic"])
        self.viz_train_semantic_gt = np.stack(
            [colour_map_np[sem] for sem in self.train_semantic_gt], axis=0
        )  # [num_test, H, W, 3]

        # scale the GT label for evaluation purpose
        self.train_semantic_gt_scaled = F.interpolate(
            torch.unsqueeze(self.train_semantic_gt, dim=1).float(),
            scale_factor=1 / self.config["render"]["test_viz_factor"],
            mode="nearest",
        ).squeeze(1)
        self.train_semantic_gt_scaled = self.train_semantic_gt_scaled.cpu().numpy() - 1

        # pose
        self.train_Ts = torch.from_numpy(train_samples["T_wc"]).float()

        #####test data#####
        # rgb
        self.test_image = torch.from_numpy(test_samples["image"])  # [num_test, H, W, 3]
        # scale the test image for evaluation purpose
        self.test_image_scaled = F.interpolate(
            self.test_image.permute(
                0,
                3,
                1,
                2,
            ),
            scale_factor=1 / self.config["render"]["test_viz_factor"],
            mode="bilinear",
        ).permute(0, 2, 3, 1)
        # depth
        self.test_depth = torch.from_numpy(test_samples["depth"])  # [num_test, H, W]
        self.viz_test_depth = np.stack(
            [
                depth2rgb(dep, min_value=self.near, max_value=self.far)
                for dep in test_samples["depth"]
            ],
            axis=0,
        )  # [num_test, H, W, 3]
        self.test_depth_scaled = (
            F.interpolate(
                torch.unsqueeze(self.test_depth, dim=1).float(),
                scale_factor=1 / self.config["render"]["test_viz_factor"],
                mode="bilinear",
            )
            .squeeze(1)
            .cpu()
            .numpy()
        )

        # semantic
        self.test_semantic = torch.from_numpy(
            test_samples["cnn_semantic"]
        )  # [num_test, H, W]
        self.viz_test_semantic = np.stack(
            [colour_map_np[sem] for sem in self.test_semantic], axis=0
        )  # [num_test, H, W, 3]

        # evaluate against CNN predictions
        self.test_semantic_scaled = F.interpolate(
            torch.unsqueeze(self.test_semantic, dim=1).float(),
            scale_factor=1 / self.config["render"]["test_viz_factor"],
            mode="nearest",
        ).squeeze(1)
        self.test_semantic_scaled = self.test_semantic_scaled.cpu().numpy() - 1

        # evaluate against perfect groundtruth
        self.test_semantic_gt = torch.from_numpy(
            test_samples["gt_semantic"]
        )  # [num_test, H, W]
        self.viz_test_semantic_gt = np.stack(
            [colour_map_np[sem] for sem in self.test_semantic_gt], axis=0
        )  # [num_test, H, W, 3]

        # scale the GT label for evaluation purpose
        self.test_semantic_gt_scaled = F.interpolate(
            torch.unsqueeze(self.test_semantic_gt, dim=1).float(),
            scale_factor=1 / self.config["render"]["test_viz_factor"],
            mode="nearest",
        ).squeeze(1)
        self.test_semantic_gt_scaled = self.test_semantic_gt_scaled.cpu().numpy() - 1

        # pose
        self.test_Ts = torch.from_numpy(
            test_samples["T_wc"]
        ).float()  # [num_test, 4, 4]

        if gpu is True:
            self.train_image = self.train_image.cuda()
            self.train_image_scaled = self.train_image_scaled.cuda()
            self.train_depth = self.train_depth.cuda()
            self.train_semantic = self.train_semantic.cuda()

            self.test_image = self.test_image.cuda()
            self.test_image_scaled = self.test_image_scaled.cuda()
            self.test_depth = self.test_depth.cuda()
            self.test_semantic = self.test_semantic.cuda()

            self.colour_map = self.colour_map.cuda()
            self.valid_colour_map = self.valid_colour_map.cuda()

        # set the data sampling paras which need the number of training images
        if (
            self.no_batching is False
        ):  # False means we need to sample from all rays instead of rays from one random image
            self.i_batch = 0
            self.rand_idx = torch.randperm(self.num_train * self.H * self.W)

        # add datasets to tfboard for comparison to rendered images
        self.tfb_viz.tb_writer.add_image(
            "Train/legend",
            np.expand_dims(legend_img_arr, axis=0),
            0,
            dataformats="NHWC",
        )
        self.tfb_viz.tb_writer.add_image(
            "Train/rgb_GT", train_samples["image"], 0, dataformats="NHWC"
        )
        self.tfb_viz.tb_writer.add_image(
            "Train/depth_GT", self.viz_train_depth, 0, dataformats="NHWC"
        )
        self.tfb_viz.tb_writer.add_image(
            "Train/vis_CNN_sem_label", self.viz_train_semantic, 0, dataformats="NHWC"
        )
        self.tfb_viz.tb_writer.add_image(
            "Train/vis_CNN_sem_label_clean",
            self.viz_train_semantic_clean,
            0,
            dataformats="NHWC",
        )
        self.tfb_viz.tb_writer.add_image(
            "Train/vis_GT_sem_label", self.viz_train_semantic_gt, 0, dataformats="NHWC"
        )

        self.tfb_viz.tb_writer.add_image(
            "Test/legend", np.expand_dims(legend_img_arr, axis=0), 0, dataformats="NHWC"
        )
        self.tfb_viz.tb_writer.add_image(
            "Test/rgb_GT", test_samples["image"], 0, dataformats="NHWC"
        )
        self.tfb_viz.tb_writer.add_image(
            "Test/depth_GT", self.viz_test_depth, 0, dataformats="NHWC"
        )
        self.tfb_viz.tb_writer.add_image(
            "Test/vis_CNN_sem_label", self.viz_test_semantic, 0, dataformats="NHWC"
        )
        self.tfb_viz.tb_writer.add_image(
            "Test/vis_GT_sem_label", self.viz_test_semantic_gt, 0, dataformats="NHWC"
        )

    def prepare_data_scannet(self, data, gpu=True):
        self.ignore_label = -1

        # shift numpy data to torch
        train_samples = data.train_samples
        test_samples = data.test_samples

        self.train_ids = data.train_ids
        self.test_ids = data.test_ids
        self.mask_ids = data.mask_ids

        self.num_train = data.train_num
        self.num_test = data.test_num

        # preprocess semantic info
        self.semantic_classes = torch.from_numpy(data.semantic_classes)
        self.num_semantic_class = self.semantic_classes.shape[
            0
        ]  # number of semantic classes, including void class=0
        self.num_valid_semantic_class = (
            self.num_semantic_class - 1
        )  # exclude void class ==0
        assert self.num_semantic_class == data.num_semantic_class

        colour_map_np = data.colour_map_np_remap
        self.colour_map = torch.from_numpy(colour_map_np)
        self.valid_colour_map = torch.from_numpy(
            colour_map_np[1:, :]
        )  # exclude the first colour map to colourise rendered segmentation without void index

        # plot semantic label legend
        class_name_string = [
            "void",
            "wall",
            "floor",
            "cabinet",
            "bed",
            "chair",
            "sofa",
            "table",
            "door",
            "window",
            "book",
            "picture",
            "counter",
            "blinds",
            "desk",
            "shelves",
            "curtain",
            "dresser",
            "pillow",
            "mirror",
            "floor",
            "clothes",
            "ceiling",
            "books",
            "fridge",
            "tv",
            "paper",
            "towel",
            "shower curtain",
            "box",
            "white board",
            "person",
            "night stand",
            "toilet",
            "sink",
            "lamp",
            "bath tub",
            "bag",
            "other struct",
            "other furntr",
            "other prop",
        ]  # NYUv2-40-class

        legend_img_arr = image_utils.plot_semantic_legend(
            data.semantic_classes,
            class_name_string,
            colormap=data.colour_map_np,
            save_path=self.save_dir,
        )
        # total_num_classes +1 to include void class

        # remap different semantic classes to continuous integers from 0 to num_class-1
        self.semantic_classes_remap = torch.from_numpy(
            np.arange(self.num_semantic_class)
        )

        #####training data#####
        # rgb
        self.train_image = torch.from_numpy(train_samples["image"])
        self.train_image_scaled = F.interpolate(
            self.train_image.permute(
                0,
                3,
                1,
                2,
            ),
            scale_factor=1 / self.config["render"]["test_viz_factor"],
            mode="bilinear",
        ).permute(0, 2, 3, 1)
        # depth
        self.train_depth = torch.from_numpy(train_samples["depth"])
        self.viz_train_depth = np.stack(
            [
                depth2rgb(dep, min_value=self.near, max_value=self.far)
                for dep in train_samples["depth"]
            ],
            axis=0,
        )  # [num_test, H, W, 3]
        self.train_depth_scaled = (
            F.interpolate(
                torch.unsqueeze(self.train_depth, dim=1).float(),
                scale_factor=1 / self.config["render"]["test_viz_factor"],
                mode="bilinear",
            )
            .squeeze(1)
            .cpu()
            .numpy()
        )

        # semantic
        self.train_semantic = torch.from_numpy(train_samples["semantic_remap"])
        self.viz_train_semantic = np.stack(
            [colour_map_np[sem] for sem in self.train_semantic], axis=0
        )  # [num_test, H, W, 3]

        self.train_semantic_clean = torch.from_numpy(
            train_samples["semantic_remap_clean"]
        )
        self.viz_train_semantic_clean = np.stack(
            [colour_map_np[sem] for sem in self.train_semantic_clean], axis=0
        )  # [num_test, H, W, 3]

        # process the clean label for evaluation purpose
        self.train_semantic_clean_scaled = F.interpolate(
            torch.unsqueeze(self.train_semantic_clean, dim=1).float(),
            scale_factor=1 / self.config["render"]["test_viz_factor"],
            mode="nearest",
        ).squeeze(1)
        self.train_semantic_clean_scaled = (
            self.train_semantic_clean_scaled.cpu().numpy() - 1
        )
        # pose
        self.train_Ts = torch.from_numpy(train_samples["T_wc"]).float()

        #####test data#####
        # rgb
        self.test_image = torch.from_numpy(test_samples["image"])  # [num_test, H, W, 3]
        # scale the test image for evaluation purpose
        self.test_image_scaled = F.interpolate(
            self.test_image.permute(
                0,
                3,
                1,
                2,
            ),
            scale_factor=1 / self.config["render"]["test_viz_factor"],
            mode="bilinear",
        ).permute(0, 2, 3, 1)

        # depth
        self.test_depth = torch.from_numpy(test_samples["depth"])  # [num_test, H, W]
        self.viz_test_depth = np.stack(
            [
                depth2rgb(dep, min_value=self.near, max_value=self.far)
                for dep in test_samples["depth"]
            ],
            axis=0,
        )  # [num_test, H, W, 3]
        self.test_depth_scaled = (
            F.interpolate(
                torch.unsqueeze(self.test_depth, dim=1).float(),
                scale_factor=1 / self.config["render"]["test_viz_factor"],
                mode="bilinear",
            )
            .squeeze(1)
            .cpu()
            .numpy()
        )

        # semantic
        self.test_semantic = torch.from_numpy(
            test_samples["semantic_remap"]
        )  # [num_test, H, W]
        # self.viz_test_semantic = torch.cat([self.colour_map[sem] for sem in self.test_semantic], dim=0).numpy() # [num_test, H, W, 3]
        self.viz_test_semantic = np.stack(
            [colour_map_np[sem] for sem in self.test_semantic], axis=0
        )  # [num_test, H, W, 3]

        # we do add noise only to training images used for training, test images are kept the same. No need for test_remap_clean

        # process the clean label for evaluation purpose
        self.test_semantic_scaled = F.interpolate(
            torch.unsqueeze(self.test_semantic, dim=1).float(),
            scale_factor=1 / self.config["render"]["test_viz_factor"],
            mode="nearest",
        ).squeeze(1)
        self.test_semantic_scaled = (
            self.test_semantic_scaled.cpu().numpy() - 1
        )  # shift void class from value 0 to -1, to match self.ignore_label
        # pose
        self.test_Ts = torch.from_numpy(
            test_samples["T_wc"]
        ).float()  # [num_test, 4, 4]

        if gpu is True:
            self.train_image = self.train_image.cuda()
            self.train_image_scaled = self.train_image_scaled.cuda()
            self.train_depth = self.train_depth.cuda()
            self.train_semantic = self.train_semantic.cuda()

            self.test_image = self.test_image.cuda()
            self.test_image_scaled = self.test_image_scaled.cuda()
            self.test_depth = self.test_depth.cuda()
            self.test_semantic = self.test_semantic.cuda()
            self.colour_map = self.colour_map.cuda()
            self.valid_colour_map = self.valid_colour_map.cuda()

        # set the data sampling paras which need the number of training images
        if (
            self.no_batching is False
        ):  # False means we need to sample from all rays instead of rays from one random image
            self.i_batch = 0
            self.rand_idx = torch.randperm(self.num_train * self.H * self.W)

        # add datasets to tfboard for comparison to rendered images
        self.tfb_viz.tb_writer.add_image(
            "Train/legend",
            np.expand_dims(legend_img_arr, axis=0),
            0,
            dataformats="NHWC",
        )
        self.tfb_viz.tb_writer.add_image(
            "Train/rgb_GT", train_samples["image"], 0, dataformats="NHWC"
        )
        self.tfb_viz.tb_writer.add_image(
            "Train/depth_GT", self.viz_train_depth, 0, dataformats="NHWC"
        )
        self.tfb_viz.tb_writer.add_image(
            "Train/vis_sem_label_GT", self.viz_train_semantic, 0, dataformats="NHWC"
        )
        self.tfb_viz.tb_writer.add_image(
            "Train/vis_sem_label_GT_clean",
            self.viz_train_semantic_clean,
            0,
            dataformats="NHWC",
        )

        self.tfb_viz.tb_writer.add_image(
            "Test/legend", np.expand_dims(legend_img_arr, axis=0), 0, dataformats="NHWC"
        )
        self.tfb_viz.tb_writer.add_image(
            "Test/rgb_GT", test_samples["image"], 0, dataformats="NHWC"
        )
        self.tfb_viz.tb_writer.add_image(
            "Test/depth_GT", self.viz_test_depth, 0, dataformats="NHWC"
        )
        self.tfb_viz.tb_writer.add_image(
            "Test/vis_sem_label_GT", self.viz_test_semantic, 0, dataformats="NHWC"
        )

    def init_rays(self):

        # create rays
        rays = create_rays(
            self.num_train,
            self.train_Ts,
            self.H,
            self.W,
            self.fx,
            self.fy,
            self.cx,
            self.cy,
            self.near,
            self.far,
            use_viewdirs=self.use_viewdir,
            convention=self.convention,
        )
        # rays contains all the rays of every pixel in all training images
        rays_vis = create_rays(
            self.num_train,
            self.train_Ts,
            self.H_scaled,
            self.W_scaled,
            self.fx_scaled,
            self.fy_scaled,
            self.cx_scaled,
            self.cy_scaled,
            self.near,
            self.far,
            use_viewdirs=self.use_viewdir,
            convention=self.convention,
        )

        rays_test = create_rays(
            self.num_test,
            self.test_Ts,
            self.H_scaled,
            self.W_scaled,
            self.fx_scaled,
            self.fy_scaled,
            self.cx_scaled,
            self.cy_scaled,
            self.near,
            self.far,
            use_viewdirs=self.use_viewdir,
            convention=self.convention,
        )

        # init rays
        self.rays = rays.cuda()  # [num_images, H*W, 11]
        self.rays_vis = rays_vis.cuda()
        self.rays_test = rays_test.cuda()

    def sample_data(self, rays, h, w, no_batching=True, mode="train"):
        # generate sampling index
        num_img, num_ray, ray_dim = rays.shape
        assert num_ray == h * w
        total_ray_num = num_img * h * w

        if mode == "train":
            image = self.train_image
            depth = self.train_depth
            surface_normal = self.train_surface_normal
            semantic = self.train_semantic
            sample_num = self.num_train
        elif mode == "test":
            image = self.test_image
            depth = self.test_depth
            surface_normal = self.test_surface_normal
            semantic = self.test_semantic
            sample_num = self.num_test
        elif mode == "vis":
            assert False
        else:
            assert False

        # sample rays and ground truth data
        semantic_available_flag = 1

        if no_batching:  # sample random pixels from one random images
            batch_index, hw_index = sampling_index(self.n_rays, num_img, h, w)
            sampled_rays = rays[
                batch_index, hw_index, :
            ]  # sample some rays of pixels from a single image
            flat_sampled_rays = sampled_rays.reshape([-1, ray_dim]).float()
            gt_image = image.reshape(sample_num, -1, 3)[
                batch_index, hw_index, :
            ].reshape(-1, 3)
            gt_depth = depth.reshape(sample_num, -1)[batch_index, hw_index].reshape(-1)
            gt_surface_normal = surface_normal.reshape(sample_num, -1, 3)[
                batch_index, hw_index, :
            ].reshape(-1, 3)
            semantic_available_flag = self.mask_ids[
                batch_index
            ]  # semantic available if mask_id is 1 (train with rgb loss and semantic loss) else 0 (train with rgb loss only)
            gt_semantic = semantic.reshape(sample_num, -1)[
                batch_index, hw_index
            ].reshape(-1)
            gt_semantic = gt_semantic.cuda()
        else:  # sample from all random pixels

            hw_index = self.rand_idx[self.i_batch : self.i_batch + self.n_rays]

            flat_rays = rays.reshape([-1, ray_dim]).float()
            flat_sampled_rays = flat_rays[hw_index, :]
            gt_image = image.reshape(-1, 3)[hw_index, :]
            if self.enable_multitask:
                gt_depth = depth.reshape(-1)[hw_index]
                gt_surface_normal = surface_normal.reshape(-1, 3)[hw_index, :]
                gt_semantic = semantic.reshape(-1)[hw_index]
                gt_semantic = gt_semantic.cuda()

            self.i_batch += self.n_rays
            if self.i_batch >= total_ray_num:
                print("Shuffle data after an epoch!")
                self.rand_idx = torch.randperm(total_ray_num)
                self.i_batch = 0

        sampled_rays = flat_sampled_rays
        sampled_gt_rgb = gt_image
        if self.enable_multitask:
            sampled_gt_depth = gt_depth
            sampled_gt_surface_normal = gt_surface_normal
            sampled_gt_semantic = (
                gt_semantic.long()
            )  # required long type for nn.NLL or nn.crossentropy

            return (
                sampled_rays,
                sampled_gt_rgb,
                sampled_gt_depth,
                sampled_gt_surface_normal,
                sampled_gt_semantic,
                semantic_available_flag,
            )
        else:
            return sampled_rays, sampled_gt_rgb

    def render_rays(self, flat_rays):
        """
        Render rays, run in optimisation loop
        Returns:
          List of:
            rgb_map: [batch_size, 3]. Predicted RGB values for rays.
            disp_map: [batch_size]. Disparity map. Inverse of depth.
            acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
          Dict of extras: dict with everything returned by render_rays().
        """

        # Render and reshape
        ray_shape = flat_rays.shape  # [num_rays, 11]

        # assert ray_shape[0] == self.n_rays  # this is not satisfied in test model
        fn = self.volumetric_rendering
        all_ret = batchify_rays(fn, flat_rays, self.chunk)

        for key in all_ret:
            k_sh = list(ray_shape[:-1]) + list(
                all_ret[key].shape[1:]
            )  # [num_rays, ...]
            all_ret[key] = torch.reshape(all_ret[key], k_sh)

        return all_ret

    def render_rays_chunk(self, flat_rays, chunk_size=1024 * 4):
        """
        Render rays while moving resulting chunks to cpu to avoid OOM when rendering large images.
        Only used in render_path. Not used in optimization.
        """
        B = flat_rays.shape[0]  # num_rays
        results = defaultdict(list)
        for i in range(0, B, chunk_size):
            rendered_ray_chunks = self.render_rays(
                flat_rays[i : i + chunk_size]
            )  # render_rays returns a dict

            for key, value in rendered_ray_chunks.items():
                results[key] += [value.cpu()]

        for key, value in results.items():
            results[key] = torch.cat(value, 0)
        return results

    def volumetric_rendering(self, ray_batch):
        """
        Volumetric Rendering, each ray is a vector of 11 elements
        """
        N_rays = ray_batch.shape[0]

        rays_o, rays_d = (
            ray_batch[:, 0:3],
            ray_batch[:, 3:6],
        )  # [N_rays, 3] each, 6 elems
        viewdirs = ray_batch[:, -3:] if ray_batch.shape[-1] > 8 else None  # 3 elems

        bounds = torch.reshape(ray_batch[..., 6:8], [-1, 1, 2])  # 2 elems
        near, far = bounds[..., 0], bounds[..., 1]  # [N_rays, 1], [N_rays, 1]

        t_vals = torch.linspace(0.0, 1.0, steps=self.N_samples).cuda()  # [N_samples]

        z_vals = near * (1.0 - t_vals) + far * (
            t_vals
        )  # use linear sampling in depth space, uniformly distributed between near and far bound
        z_vals = z_vals.expand([N_rays, self.N_samples])

        if self.perturb > 0.0 and self.training:  # perturb sampling depths (z_vals)
            if (
                self.training is True
            ):  # only add perturbation during training instead of testing
                # get intervals between samples
                mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
                upper = torch.cat([mids, z_vals[..., -1:]], -1)
                lower = torch.cat([z_vals[..., :1], mids], -1)
                # stratified samples in those intervals
                t_rand = torch.rand(z_vals.shape).cuda()

                z_vals = lower + (upper - lower) * t_rand

        pts_coarse_sampled = (
            rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
        )  # [N_rays, N_samples, 3]

        raw_noise_std = self.raw_noise_std if self.training else 0
        raw_coarse = run_network(
            pts_coarse_sampled,
            viewdirs,
            self.ssr_net_coarse,
            self.embed_fn,
            self.embeddirs_fn,
            netchunk=self.netchunk,
        )
        (
            rgb_coarse,
            disp_coarse,
            acc_coarse,
            weights_coarse,
            depth_coarse,
            surface_normal_coarse,
            sem_logits_coarse,
            feat_map_coarse,
        ) = raw2outputs(
            raw_coarse,
            z_vals,
            rays_d,
            raw_noise_std,
            self.white_bkgd,
            enable_multitask=self.enable_multitask,
            num_sem_class=self.num_valid_semantic_class,
            endpoint_feat=False,
        )  # whether enable_multitask of not, output of raw2outputs has the same number

        if self.N_importance > 0:
            z_vals_mid = 0.5 * (
                z_vals[..., 1:] + z_vals[..., :-1]
            )  # (N_rays, N_samples-1) interval mid points
            z_samples = sample_pdf(
                z_vals_mid,
                weights_coarse[..., 1:-1],
                self.N_importance,
                det=(self.perturb == 0.0) or (not self.training),
            )
            z_samples = z_samples.detach()
            # detach so that grad doesn't propagate to weights_coarse from here
            # values are interleaved actually, so maybe can do better than sort?

            z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
            pts_fine_sampled = (
                rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
            )  # [N_rays, N_samples + N_importance, 3]

            raw_fine = run_network(
                pts_fine_sampled,
                viewdirs,
                lambda x: self.ssr_net_fine(x, self.endpoint_feat),
                self.embed_fn,
                self.embeddirs_fn,
                netchunk=self.netchunk,
            )

            (
                rgb_fine,
                disp_fine,
                acc_fine,
                weights_fine,
                depth_fine,
                surface_normal_fine,
                sem_logits_fine,
                feat_map_fine,
            ) = raw2outputs(
                raw_fine,
                z_vals,
                rays_d,
                raw_noise_std,
                self.white_bkgd,
                enable_multitask=self.enable_multitask,
                num_sem_class=self.num_valid_semantic_class,
                endpoint_feat=self.endpoint_feat,
            )

        ret = {}
        ret["raw_coarse"] = raw_coarse
        ret["rgb_coarse"] = rgb_coarse
        ret["disp_coarse"] = disp_coarse
        ret["acc_coarse"] = acc_coarse
        ret["depth_coarse"] = depth_coarse
        ret["surface_normal_coarse"] = surface_normal_coarse
        if self.enable_multitask:
            ret["sem_logits_coarse"] = sem_logits_coarse

        if self.N_importance > 0:
            ret["raw_fine"] = raw_fine  # model's raw, unprocessed predictions.
            ret["rgb_fine"] = rgb_fine
            ret["disp_fine"] = disp_fine
            ret["acc_fine"] = acc_fine
            ret["depth_fine"] = depth_fine
            ret["surface_normal_fine"] = surface_normal_fine
            if self.enable_multitask:
                ret["sem_logits_fine"] = sem_logits_fine
            ret["z_std"] = torch.std(z_samples, dim=-1, unbiased=False)  # [N_rays]
            if self.endpoint_feat:
                ret["feat_map_fine"] = feat_map_fine
        for key in ret:
            # if (torch.isnan(ret[key]).any() or torch.isinf(ret[key]).any()) and self.config["experiment"]["debug"]:
            if torch.isnan(ret[key]).any() or torch.isinf(ret[key]).any():
                print(f"! [Numerical Error] {key} contains nan or inf.")

        return ret

    def create_ssr(self):
        """Instantiate NeRF's MLP model."""

        nerf_model = Semantic_NeRF

        embed_fn, input_ch = get_embedder(
            self.config["render"]["multires"],
            self.config["render"]["i_embed"],
            scalar_factor=10,
        )

        input_ch_views = 0
        embeddirs_fn = None
        if self.config["render"]["use_viewdirs"]:
            embeddirs_fn, input_ch_views = get_embedder(
                self.config["render"]["multires_views"],
                self.config["render"]["i_embed"],
                scalar_factor=1,
            )
        output_ch = 5 if self.N_importance > 0 else 4
        skips = [4]
        model = nerf_model(
            enable_multitask=self.enable_multitask,
            enable_semantic=self.enable_semantic,
            enable_surface_normal=self.enable_surface_normal,
            num_semantic_classes=self.num_valid_semantic_class,
            d=self.config["model"]["netdepth"],
            w=self.config["model"]["netwidth"],
            input_ch=input_ch,
            output_ch=output_ch,
            skips=skips,
            input_ch_views=input_ch_views,
            use_viewdirs=self.config["render"]["use_viewdirs"],
        ).cuda()
        grad_vars = list(model.parameters())

        model_fine = None
        if self.N_importance > 0:
            model_fine = nerf_model(
                enable_multitask=self.enable_multitask,
                enable_semantic=self.enable_semantic,
                enable_surface_normal=self.enable_surface_normal,
                num_semantic_classes=self.num_valid_semantic_class,
                d=self.config["model"]["netdepth_fine"],
                w=self.config["model"]["netwidth_fine"],
                input_ch=input_ch,
                output_ch=output_ch,
                skips=skips,
                input_ch_views=input_ch_views,
                use_viewdirs=self.config["render"]["use_viewdirs"],
            ).cuda()
            grad_vars += list(model_fine.parameters())

        # Create optimizer
        # optimizer = torch.optim.Adam(params=grad_vars, lr=self.lrate)
        optimizer = optim.Lamb(params=grad_vars)

        self.ssr_net_coarse = model
        self.ssr_net_fine = model_fine
        self.embed_fn = embed_fn
        self.embeddirs_fn = embeddirs_fn
        self.optimizer = optimizer

    # optimisation step
    def step(self, global_step):
        # Misc
        img2mse = lambda x, y: torch.mean((x - y) ** 2)
        self.global_step = global_step
        mse2psnr = (
            lambda x: -10.0 * torch.log(x) / torch.log(torch.Tensor([10.0]).cuda())
        )
        CrossEntropyLoss = nn.CrossEntropyLoss(ignore_index=self.ignore_label)
        KLDLoss = nn.KLDivLoss(reduction="none")
        kl_loss = lambda input_log_prob, target_prob: KLDLoss(
            input_log_prob, target_prob
        )
        # this function assume input is already in log-probabilities

        dataset_type = self.config["experiment"]["dataset_type"]
        if (
            dataset_type == "replica"
            or dataset_type == "replica_nyu_cnn"
            or dataset_type == "scannet"
        ):
            crossentropy_loss = lambda logit, label: CrossEntropyLoss(
                logit, label - 1
            )  # replica has void class of ID==0, label-1 to shift void class to -1
        else:
            assert False

        logits_2_label = lambda x: torch.argmax(
            torch.nn.functional.softmax(x, dim=-1), dim=-1
        )
        logits_2_prob = lambda x: F.softmax(x, dim=-1)
        to8b_np = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)
        to8b = lambda x: (255 * torch.clamp(x, 0, 1)).type(torch.uint8)

        # sample rays to query and optimise
        """
        where to get data as input for the model
        """
        sampled_data = self.sample_data(
            self.rays, self.H, self.W, no_batching=True, mode="train"
        )
        if self.enable_multitask:
            (
                sampled_rays,
                sampled_gt_rgb,
                sampled_gt_depth,
                sampled_surface_normal,
                sampled_gt_semantic,
                semantic_available,
            ) = sampled_data
        else:
            sampled_rays, sampled_gt_rgb = sampled_data

        output_dict = self.render_rays(sampled_rays)

        rgb_coarse = output_dict["rgb_coarse"]  # N_rays x 3
        disp_coarse = output_dict["disp_coarse"]  # N_rays
        depth_coarse = output_dict["depth_coarse"]  # N_rays
        acc_coarse = output_dict["acc_coarse"]  # N_rays
        if self.enable_multitask:
            sem_logits_coarse = output_dict["sem_logits_coarse"]  # N_rays x num_classes
            sem_label_coarse = logits_2_label(sem_logits_coarse)  # N_rays
            surface_normal_coarse = output_dict["surface_normal_coarse"]
        if self.N_importance > 0:
            rgb_fine = output_dict["rgb_fine"]
            disp_fine = output_dict["disp_fine"]
            depth_fine = output_dict["depth_fine"]
            acc_fine = output_dict["acc_fine"]
            z_std = output_dict["z_std"]  # N_rays
            if self.enable_multitask:
                sem_logits_fine = output_dict["sem_logits_fine"]
                sem_label_fine = logits_2_label(sem_logits_fine)
                surface_normal_fine = output_dict["surface_normal_fine"]

        self.optimizer.zero_grad()

        img_loss_coarse = F.mse_loss(rgb_coarse, sampled_gt_rgb)

        if self.enable_multitask and semantic_available:
            semantic_loss_coarse = crossentropy_loss(
                sem_logits_coarse, sampled_gt_semantic
            )
            depth_loss_coarse = F.mse_loss(depth_coarse, sampled_gt_depth)
            surface_normal_loss_coarse = F.cosine_similarity(
                surface_normal_coarse, sampled_surface_normal
            ).mean()
        else:
            semantic_loss_coarse = torch.tensor(0)
            depth_loss_coarse = torch.tensor(0)
            surface_normal_loss_coarse = torch.tensor(0)

        with torch.no_grad():
            psnr_coarse = mse2psnr(img_loss_coarse)
            self.psnr_coarse.append(psnr_coarse.item())

        if self.N_importance > 0:
            img_loss_fine = F.mse_loss(rgb_fine, sampled_gt_rgb)
            if self.enable_multitask and semantic_available:
                semantic_loss_fine = (
                    crossentropy_loss(sem_logits_fine, sampled_gt_semantic)
                    if self.enable_semantic
                    else torch.tensor(0)
                )
                depth_loss_fine = (
                    F.mse_loss(depth_fine, sampled_gt_depth)
                    if self.enable_depth
                    else torch.tensor(0)
                )
                surface_normal_loss_fine = (
                    F.cosine_similarity(
                        surface_normal_fine, sampled_surface_normal
                    ).mean()
                    if self.enable_surface_normal
                    else torch.tensor(0)
                )
            else:
                semantic_loss_fine = torch.tensor(0)
                depth_loss_fine = torch.tensor(0)
                surface_normal_loss_fine = torch.tensor(0)
            with torch.no_grad():
                psnr_fine = mse2psnr(img_loss_fine)
                self.psnr_fine.append(psnr_fine.item())
        else:
            img_loss_fine = torch.tensor(0)
            semantic_loss_fine = torch.tensor(0)
            depth_loss_fine = torch.tensor(0)
            surface_normal_loss_fine = torch.tensor(0)
            psnr_fine = torch.tensor(0)
            self.psnr_fine.append(psnr_fine)

        total_img_loss = img_loss_coarse + img_loss_fine
        total_sem_loss = (
            semantic_loss_coarse + semantic_loss_fine
            if self.enable_semantic
            else torch.tensor(0)
        )
        total_depth_loss = (
            depth_loss_coarse + depth_loss_fine
            if self.enable_depth
            else torch.tensor(0)
        )
        total_norm_loss = (
            surface_normal_loss_coarse + surface_normal_loss_fine
            if self.enable_surface_normal
            else torch.tensor(0)
        )
        self.img_loss.append(img_loss_fine.item())
        self.sem_loss.append(semantic_loss_fine.item())
        self.depth_loss.append(depth_loss_fine.item())
        self.norm_loss.append(surface_normal_loss_fine.item())

        wgt_sem_loss = float(self.config["train"]["wgt_sem"])
        wgt_depth_loss = float(self.config["train"]["wgt_depth"])
        wgt_norm_loss = float(self.config["train"]["wgt_norm"])
        total_loss = total_img_loss
        if self.enable_multitask:
            if self.enable_semantic:
                total_loss += total_sem_loss * wgt_sem_loss
            if self.enable_surface_normal:
                total_loss -= total_norm_loss * wgt_norm_loss
        total_loss += total_depth_loss * wgt_depth_loss
        total_loss.backward()
        self.optimizer.step()

        ###   update learning rate   ###
        # decay_rate = 0.1
        # decay_steps = self.lrate_decay
        # new_lrate = self.lrate * (decay_rate ** (global_step / decay_steps))
        # for param_group in self.optimizer.param_groups:
        #     param_group["lr"] = new_lrate

    def test(self):
        mse2psnr = lambda x: (
            -10.0 * torch.log(x) / torch.log(torch.Tensor([10.0]).cuda())
        ).squeeze()
        sampled_data = self.sample_data(
            self.rays_test, self.H, self.W, no_batching=True, mode="test"
        )
        if self.enable_multitask:
            (
                sampled_rays,
                sampled_gt_rgb,
                sampled_gt_depth,
                sampled_surface_normal,
                sampled_gt_semantic,
                semantic_available,
            ) = sampled_data
        else:
            sampled_rays, sampled_gt_rgb = sampled_data

        with torch.no_grad():
            output_dict = self.render_rays(sampled_rays)

        rgb_coarse = output_dict["rgb_coarse"]  # N_rays x 3
        disp_coarse = output_dict["disp_coarse"]  # N_rays
        depth_coarse = output_dict["depth_coarse"]  # N_rays
        acc_coarse = output_dict["acc_coarse"]  # N_rays
        if self.enable_multitask:
            sem_logits_coarse = output_dict["sem_logits_coarse"]  # N_rays x num_classes
            # sem_label_coarse = logits_2_label(sem_logits_coarse)  # N_rays
            surface_normal_coarse = output_dict["surface_normal_coarse"]
        if self.N_importance > 0:
            rgb_fine = output_dict["rgb_fine"]
            disp_fine = output_dict["disp_fine"]
            depth_fine = output_dict["depth_fine"]
            acc_fine = output_dict["acc_fine"]
            z_std = output_dict["z_std"]  # N_rays
            if self.enable_multitask:
                sem_logits_fine = output_dict["sem_logits_fine"]
                # sem_label_fine = logits_2_label(sem_logits_fine)
                surface_normal_fine = output_dict["surface_normal_fine"]
        img_loss_coarse = F.mse_loss(rgb_coarse, sampled_gt_rgb)
        if self.enable_multitask and semantic_available:
            semantic_loss_coarse = F.cross_entropy(
                sem_logits_coarse,
                sampled_gt_semantic - 1,
                ignore_index=self.ignore_label,
            )
            depth_loss_coarse = F.mse_loss(depth_coarse, sampled_gt_depth)
            surface_normal_loss_coarse = F.cosine_similarity(
                surface_normal_coarse, sampled_surface_normal
            ).mean()
            iou_coarse, _, _, _, _ = calculate_segmentation_metrics(
                sampled_gt_semantic.cpu() - 1,
                sem_logits_coarse.softmax(dim=-1).argmax(dim=-1).cpu(),
                self.num_valid_semantic_class,
                self.ignore_label,
            )
        else:
            semantic_loss_coarse = torch.tensor(0)
            depth_loss_coarse = torch.tensor(0)
            surface_normal_loss_coarse = torch.tensor(0)
            iou_coarse = torch.tensor(0)

        with torch.no_grad():
            psnr_coarse = mse2psnr(img_loss_coarse)

        if self.N_importance > 0:
            img_loss_fine = F.mse_loss(rgb_fine, sampled_gt_rgb)
            if self.enable_multitask and semantic_available:
                semantic_loss_fine = F.cross_entropy(
                    sem_logits_fine,
                    sampled_gt_semantic - 1,
                    ignore_index=self.ignore_label,
                )
                depth_loss_fine = F.mse_loss(depth_fine, sampled_gt_depth)
                surface_normal_loss_fine = F.cosine_similarity(
                    surface_normal_fine, sampled_surface_normal
                ).mean()
                iou_fine, _, _, _, _ = calculate_segmentation_metrics(
                    sampled_gt_semantic.cpu() - 1,
                    sem_logits_fine.softmax(dim=-1).argmax(dim=-1).cpu(),
                    self.num_valid_semantic_class,
                    self.ignore_label,
                )
            else:
                semantic_loss_fine = torch.tensor(0)
                depth_loss_fine = torch.tensor(0)
                surface_normal_loss_fine = torch.tensor(0)
                iou_fine = torch.tensor(0)
            with torch.no_grad():
                psnr_fine = mse2psnr(img_loss_fine)
        else:
            img_loss_fine = torch.tensor(0)
            psnr_fine = torch.tensor(0)
            self.psnr_fine.append(psnr_fine)

        total_img_loss = img_loss_coarse + img_loss_fine
        total_sem_loss = semantic_loss_coarse + semantic_loss_fine
        total_depth_loss = depth_loss_coarse + depth_loss_fine
        total_norm_loss = surface_normal_loss_coarse + surface_normal_loss_fine

        rich.print(
            "image psnr: {:.4f}, IoU: {:.4f}, surface normal similarity: {:.6f}, depth loss: {:.6f}".format(
                psnr_fine,
                iou_fine if self.enable_semantic else 0,
                surface_normal_loss_fine if self.enable_surface_normal else 0,
                depth_loss_fine if self.enable_depth else 0,
            )
        )

        file = "/home/sw99/huaizhi_qu/semantic_nerf/results/{}".format(
            self.config["train"]["N_iters"]
        )
        if self.enable_multitask:
            file += "_multitask"
        if self.enable_semantic:
            file += "_semantic"
        if self.enable_depth:
            file += "_depth"
        if self.enable_surface_normal:
            file += "_norm"
        file += ".csv"
        pd.DataFrame.from_dict(
            {
                "step": [self.global_step],
                "psnr": [psnr_fine.item()],
                "IoU": [iou_fine.item()] if self.enable_semantic else [0],
                "surface normal similarity": [surface_normal_loss_fine.item()]
                if self.enable_surface_normal
                else [0],
                "depth": [depth_loss_fine.item()] if self.enable_depth else [0],
            }
        ).to_csv(
            file,
            mode="a",
            header=False if os.path.exists(file) else True,
            index=False,
        )
