import argparse
import os

import matplotlib.pyplot as plt
import yaml
import torch
import numpy as np
from pytorch_lightning import seed_everything
from tqdm import trange
from rich.progress import (
    track,
    Progress,
    SpinnerColumn,
    TimeElapsedColumn,
    MofNCompleteColumn,
    TextColumn,
    BarColumn,
    TimeRemainingColumn,
    TaskProgressColumn,
)

from SSR.datasets.replica import replica_datasets

# from SSR.datasets.scannet import scannet_datasets
from SSR.datasets.replica_nyu import replica_nyu_cnn_datasets
from SSR.datasets.scannet import scannet_datasets
from SSR.training import trainer


def train():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--config_file', type=str, default="/home/shuaifeng/Documents/PhD_Research/CodeRelease/SemanticSceneRepresentations/SSR/configs/SSR_room2_config_release.yaml",
    #                     help='config file name.')
    parser.add_argument(
        "--config_file",
        type=str,
        default="/home/sw99/huaizhi_qu/semantic_nerf/SSR/configs/SSR_Replica_config.yaml",
        help="config file name.",
    )
    parser.add_argument(
        "--dataset_type",
        type=str,
        default="replica",
        choices=["replica", "replica_nyu_cnn", "scannet"],
        help="the dataset to be used,",
    )

    ### working mode and specific options

    # sparse-views
    parser.add_argument(
        "--sparse_views",
        action="store_true",
        help="Use labels from a sparse set of frames",
    )
    parser.add_argument(
        "--sparse_ratio",
        type=float,
        default=0,
        help="The portion of dropped labelling frames during training, which can be used along with all working modes.",
    )
    parser.add_argument(
        "--label_map_ids",
        nargs="*",
        type=int,
        default=[],
        help="In sparse view mode, use selected frame ids from sequences as supervision.",
    )
    parser.add_argument(
        "--random_sample",
        action="store_true",
        help="Whether to randomly/evenly sample frames from the sequence.",
    )

    # denoising---pixel-wsie
    parser.add_argument(
        "--pixel_denoising",
        action="store_true",
        help="Whether to work in pixel-denoising tasks.",
    )
    parser.add_argument(
        "--pixel_noise_ratio",
        type=float,
        default=0,
        help="In sparse view mode, if pixel_noise_ratio > 0, the percentage of pixels to be perturbed in each sampled frame  for pixel-wise denoising task..",
    )

    # denoising---region-wsie
    parser.add_argument(
        "--region_denoising",
        action="store_true",
        help="Whether to work in region-denoising tasks by flipping class labels of chair instances in Replica Room_2",
    )
    parser.add_argument(
        "--region_noise_ratio",
        type=float,
        default=0,
        help="In region-wise denoising task, region_noise_ratio is the percentage of chair instances to be perturbed in each sampled frame for region-wise denoising task.",
    )
    parser.add_argument(
        "--uniform_flip",
        action="store_true",
        help="In region-wise denoising task, whether to change chair labels uniformly or not, i.e., by ascending area ratios. This corresponds to two set-ups mentioned in the paper.",
    )
    parser.add_argument(
        "--instance_id",
        nargs="*",
        type=int,
        default=[3, 6, 7, 9, 11, 12, 13, 48],
        help="In region-wise denoising task, the chair instance ids in Replica Room_2 to be randomly perturbed. The ids of all 8 chairs are [3, 6, 7, 9, 11, 12, 13, 48]",
    )

    # super-resolution
    parser.add_argument(
        "--super_resolution",
        action="store_true",
        help="set to render synthetic data on a white bkgd (always use for dvoxels)",
    )
    parser.add_argument(
        "--dense_sr",
        action="store_true",
        help="Whether to use dense or sparse labels for SR instead of dense labels.",
    )
    parser.add_argument(
        "--sr_factor", type=int, default=8, help="Scaling factor of super-resolution."
    )

    # label propagation
    parser.add_argument(
        "--label_propagation",
        action="store_true",
        help="Label propagation using partial seed regions.",
    )
    parser.add_argument(
        "--partial_perc",
        type=float,
        default=0,
        help="0: single-click propagation; 1: using 1-percent sub-regions for label propagation, 5: using 5-percent sub-regions for label propagation",
    )

    # misc.
    parser.add_argument(
        "--visualise_save",
        action="store_true",
        help="whether to save the noisy labels into harddrive for later usage",
    )
    parser.add_argument(
        "--load_saved",
        action="store_true",
        help="use trained noisy labels for training to ensure consistency betwwen experiments",
    )
    parser.add_argument("--gpu", type=str, default="", help="GPU IDs.")

    args = parser.parse_args()
    seed_everything(42)
    np.seterr(invalid="ignore")
    # Read YAML file
    with open(args.config_file, "r") as f:
        config = yaml.safe_load(f)
    if len(args.gpu) > 0:
        config["experiment"]["gpu"] = args.gpu
    print("Experiment GPU is {}.".format(config["experiment"]["gpu"]))
    trainer.select_gpus(config["experiment"]["gpu"])
    config["experiment"].update(vars(args))
    # Cast intrinsics to right types
    ssr_trainer = trainer.SSRTrainer(config)

    if args.dataset_type == "replica":
        print("----- Replica Dataset -----")

        total_num = 900
        step = 5
        train_ids = list(range(0, total_num, step))  # use 20% images for training
        test_ids = [x + step // 2 for x in train_ids]
        # add ids to config for later saving.
        config["experiment"]["train_ids"] = train_ids
        config["experiment"]["test_ids"] = test_ids

        # Todo: like nerf, creating sprial/test poses. Make training and test poses/ids interleaved
        replica_data_loader = replica_datasets.ReplicaDatasetCache(
            data_dir=config["experiment"]["dataset_dir"],
            train_ids=train_ids,
            test_ids=test_ids,
            img_h=config["experiment"]["height"],
            img_w=config["experiment"]["width"],
        )

        print("--------------------")
        if args.super_resolution:
            print(
                "Super Resolution Mode! Dense Label Flag is {}, SR Factor is {}".format(
                    args.dense_sr, args.sr_factor
                )
            )
            replica_data_loader.super_resolve_label(
                down_scale_factor=args.sr_factor, dense_supervision=args.dense_sr
            )
        elif args.label_propagation:
            print(
                "Label Propagation Mode! Partial labelling percentage is: {} ".format(
                    args.partial_perc
                )
            )
            replica_data_loader.simulate_user_click_partial(
                perc=args.partial_perc,
                load_saved=args.load_saved,
                visualise_save=args.visualise_save,
            )
            if args.sparse_views:  # add view-point sampling to partial sampling
                print(
                    "Sparse Viewing Labels Mode under ***Patial Labelling***! Sparse Ratio is ",
                    args.sparse_ratio,
                )
                replica_data_loader.sample_label_maps(
                    sparse_ratio=args.sparse_ratio,
                    random_sample=args.random_sample,
                    load_saved=args.load_saved,
                )
        elif args.pixel_denoising:
            print("Pixel-Denoising Mode! Noise Ratio is ", args.pixel_noise_ratio)
            replica_data_loader.add_pixel_wise_noise_label(
                sparse_views=args.sparse_views,
                sparse_ratio=args.sparse_ratio,
                random_sample=args.random_sample,
                noise_ratio=args.pixel_noise_ratio,
                visualise_save=args.visualise_save,
                load_saved=args.load_saved,
            )
        elif args.region_denoising:
            print(
                "Chair Label Flipping for Region-wise Denoising, Flip ratio is {}, Uniform Sampling is {}".format(
                    args.region_noise_ratio, args.uniform_flip
                )
            )
            replica_data_loader.add_instance_wise_noise_label(
                sparse_views=args.sparse_views,
                sparse_ratio=args.sparse_ratio,
                random_sample=args.random_sample,
                flip_ratio=args.region_noise_ratio,
                uniform_flip=args.uniform_flip,
                instance_id=args.instance_id,
                load_saved=args.load_saved,
                visualise_save=args.visualise_save,
            )

        elif args.sparse_views:
            if len(args.label_map_ids) > 0:
                print("Use label maps only for selected frames, ", args.label_map_ids)
                replica_data_loader.sample_specific_labels(
                    args.label_map_ids, train_ids
                )
            else:
                print("Sparse Labels Mode! Sparsity Ratio is ", args.sparse_ratio)
                replica_data_loader.sample_label_maps(
                    sparse_ratio=args.sparse_ratio,
                    random_sample=args.random_sample,
                    load_saved=args.load_saved,
                )

        else:
            print("Standard setup with full dense supervision.")
        ssr_trainer.set_params_replica()
        ssr_trainer.prepare_data_replica(replica_data_loader)

    elif args.dataset_type == "replica_nyu_cnn":
        print("----- Replica Dataset with NYUv2-13 CNN Predictions -----")

        print(
            "Replica_nyu_cnn mode using labels from trained CNNs: {}".format(
                config["experiment"]["nyu_mode"]
            )
        )

        total_num = 900
        step = 5

        train_ids = list(range(0, total_num, step))
        test_ids = [x + step // 2 for x in train_ids]

        # add ids to config for later saving.
        config["experiment"]["train_ids"] = train_ids
        config["experiment"]["test_ids"] = test_ids

        replica_nyu_cnn_data_loader = replica_nyu_cnn_datasets.Replica_CNN_NYU(
            data_dir=config["experiment"]["dataset_dir"],
            train_ids=train_ids,
            test_ids=test_ids,
            img_h=config["experiment"]["height"],
            img_w=config["experiment"]["width"],
            nyu_mode=config["experiment"]["nyu_mode"],
            load_softmax=config["experiment"]["load_softmax"],
        )

        ssr_trainer.set_params_replica()  # we still call params of replica here since the image sources are from Replica still
        ssr_trainer.prepare_data_replica_nyu_cnn(replica_nyu_cnn_data_loader)

    elif args.dataset_type == "scannet":
        print("----- ScanNet Dataset with NYUv2-40 Conventions-----")

        print(
            "processing ScanNet scene: ",
            os.path.basename(config["experiment"]["dataset_dir"]),
        )
        # Todo: like nerf, creating sprial/test poses. Make training and test poses/ids interleaved
        scannet_data_loader = scannet_datasets.ScanNet_Dataset(
            scene_dir=config["experiment"]["dataset_dir"],
            img_h=config["experiment"]["height"],
            img_w=config["experiment"]["width"],
            sample_step=config["experiment"]["sample_step"],
            save_dir=config["experiment"]["dataset_dir"],
        )

        print("--------------------")
        if args.super_resolution:
            print(
                "Super Resolution Mode! Dense Label Flag is {}, SR Factor is {}".format(
                    args.dense_sr, args.sr_factor
                )
            )
            scannet_data_loader.super_resolve_label(
                down_scale_factor=args.sr_factor, dense_supervision=args.dense_sr
            )

        elif args.label_propagation:
            print(
                "Partial Segmentation Mode! Partial percentage is: {} ",
                args.partial_perc,
            )
            scannet_data_loader.simulate_user_click_partial(
                perc=args.partial_perc,
                load_saved=args.load_saved,
                visualise_save=args.visualise_save,
            )

        elif args.pixel_denoising:
            print("Pixel-Denoising Mode! Noise Ratio is ", args.pixel_noise_ratio)
            scannet_data_loader.add_pixel_wise_noise_label(
                sparse_views=args.sparse_views,
                sparse_ratio=args.sparse_ratio,
                random_sample=args.random_sample,
                noise_ratio=args.pixel_noise_ratio,
                visualise_save=args.visualise_save,
                load_saved=args.load_saved,
            )
        elif args.sparse_views:
            print("Sparse Viewing Labels Mode! Sparse Ratio is ", args.sparse_ratio)
            scannet_data_loader.sample_label_maps(
                sparse_ratio=args.sparse_ratio,
                random_sample=args.random_sample,
                load_saved=args.load_saved,
            )

        ssr_trainer.set_params_scannet(scannet_data_loader)
        ssr_trainer.prepare_data_scannet(scannet_data_loader)

    # Create nerf model, init optimizer
    ssr_trainer.create_ssr()
    # Create rays in world coordinates
    ssr_trainer.init_rays()

    start = 0

    N_iters = int(float(config["train"]["N_iters"])) + 1
    global_step = start
    ##########################
    print("Begin")
    #####  Training loop  #####
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        MofNCompleteColumn(),
    ) as progress:
        task = progress.add_task("Training...", total=N_iters)
        for i in range(start, N_iters):
            # torch.cuda.synchronize()
            # time0 = time.time()
            ssr_trainer.step(global_step)
            # torch.cuda.synchronize()
            # time1 = time.time()
            # dt = time1 - time0
            if global_step % 2000 == 0:
                ssr_trainer.test()
            global_step += 1
            progress.update(task, advance=1)
    file = "results/{}".format(N_iters - 1)
    if ssr_trainer.enable_multitask:
        file += "_multitask"
    if ssr_trainer.enable_semantic:
        file += "_semantic"
    if ssr_trainer.enable_depth:
        file += "_depth"
    if ssr_trainer.enable_surface_normal:
        file += "_norm"
    plt.figure()
    x = list(range(N_iters))
    plt.plot(x, ssr_trainer.img_loss, label="image loss")
    plt.plot(x, ssr_trainer.depth_loss, label="depth loss")
    plt.plot(x, ssr_trainer.norm_loss, label="surf norm similarity")
    plt.plot(x, ssr_trainer.sem_loss, label="semantic loss")
    plt.plot(x, ssr_trainer.psnr_coarse, label="psnr coarse")
    plt.plot(x, ssr_trainer.psnr_fine, label="psnr fine")
    print("done")
    plt.xlabel("Iterations")
    plt.title("Loss vs Epoch")
    plt.legend()
    plt.savefig(file + ".pdf", format="pdf")
    torch.save(ssr_trainer.ssr_net_coarse, file + "_coarse.pt")
    torch.save(ssr_trainer.ssr_net_fine, file + "_fine.pt")
    # plt.show()


if __name__ == "__main__":
    train()
