import os
import torch
import numpy as np
import torch.utils.data as data
import random
from .build import DATASETS
import open3d as o3d
import open3d
from os import listdir
import logging
import copy
from models.PoinTr import fps
import json
from SAP.src.dpsr import DPSR
import fpsample

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@DATASETS.register_module()
class crown(data.Dataset):
    def __init__(self, config):
        self.data_root = config.DATA_PATH
        self.subset = config.subset
        self.npoints = config.N_POINTS
        self.data_list_file = os.path.join(self.data_root, f"{self.subset}.json")

        print(f"[DATASET] Open file {self.data_list_file}")
        with open(self.data_list_file, "r") as f:
            data_subset = json.load(f)

        self.file_list = []
        for sample in data_subset:
            file_path = os.path.join(
                self.data_root, "DataSamples", sample, "DataFiles/"
            )
            self.file_list.append(
                {
                    "taxonomy_id": "1",  # we only have upper jaws
                    "model_id": sample,
                    "file_path": file_path,
                }
            )
        self.dpsr = DPSR(res=(128, 128, 128), sig=2)

    def _compute_psr_for_sample(self, shell_points_01, shell_normals):
        """Compute PSR grid from [0,1] normalized points"""
        points = torch.from_numpy(shell_points_01.astype(np.float32)).unsqueeze(0)
        normals = torch.from_numpy(shell_normals.astype(np.float32)).unsqueeze(0)
        with torch.no_grad():
            psr_grid = self.dpsr(points, normals)
        return psr_grid.squeeze().numpy()

    def pc_norm(self, pc):
        centroid = np.mean(pc, axis=0)
        std_pc = np.std(pc, axis=0)
        pc = (pc - centroid) / std_pc
        return pc, centroid, std_pc

    def normalize_points_mean_std(self, main, opposing, shell):

        new_context = copy.deepcopy(main)
        new_opposing = copy.deepcopy(opposing)
        new_crown = copy.deepcopy(shell)
        # new_marginline = copy.deepcopy(marginline)

        context_mean, context_std = np.mean(
            np.concatenate((main.points, opposing.points), axis=0), axis=0
        ), np.std(np.concatenate((main.points, opposing.points), axis=0), axis=0)
        # scale values
        new_context_points = (
            np.asarray(new_context.points) - context_mean
        ) / context_std
        # new_context.points = o3d.utility.Vector3dVector(new_context_points)

        # final_context = copy.deepcopy(new_context)

        new_opposing_points = (np.asarray(opposing.points) - context_mean) / context_std
        # new_opposing.points = o3d.utility.Vector3dVector(new_opposing_points)

        new_crown_points = (np.asarray(shell.points) - context_mean) / context_std
        # new_crown.points = o3d.utility.Vector3dVector(new_crown_points)
        # new_marginline_points = (np.asarray(marginline.points) - context_mean) / context_std

        return (
            new_context_points,
            new_opposing_points,
            new_crown_points,
            context_mean,
            context_std,
        )

    def __getitem__(self, idx):

        # read points
        sample = self.file_list[idx]
        # print(sample['file_path'])

        with open(
            os.path.join(
                sample["file_path"],
                "input_points.dat",
            ),
            "rb",
        ) as f:
            input_pc = np.fromfile(f, dtype=np.float32).reshape(
                -1, 6
            )  # (x, y, z, nx, ny, nz)

        with open(
            os.path.join(
                sample["file_path"],
                "input_properties.dat",
            ),
            "rb",
        ) as f:
            # (diff to prep/antagonist, upper/lower jaw, jawSegmentKind)
            # 0-dim: diff to prep/antagonist: 0 = antagonist / prep, 1 = unn4, -1 = unn2 or whatever on the other side
            # 1-dim: upper/lower jaw: 0 = upper jaw, 1 = lower jaw
            # 2-dim: jawSegmentKind: 0 = natural tooth, 1 = crown/prep
            input_properties = np.fromfile(f, dtype=np.byte)
            input_properties = input_properties.reshape(-1, 3)

        # oppposing = antagonist
        opposing = o3d.geometry.PointCloud()
        opposing_input = input_pc[input_properties[:, 1] == 1]
        fps_index = fpsample.bucket_fps_kdline_sampling(
            opposing_input[:, :3], self.npoints, h=5
        )
        opposing_input = opposing_input[fps_index]
        opposing.points = o3d.utility.Vector3dVector(opposing_input[:, :3])
        opposing.normals = o3d.utility.Vector3dVector(opposing_input[:, 3:6])

        # main = upper jaw
        main = o3d.geometry.PointCloud()
        main_input = input_pc[input_properties[:, 1] == 0]
        fps_index = fpsample.bucket_fps_kdline_sampling(
            main_input[:, :3], self.npoints, h=5
        )
        main_input = main_input[fps_index]
        main.points = o3d.utility.Vector3dVector(main_input[:, :3])
        main.normals = o3d.utility.Vector3dVector(main_input[:, 3:6])

        shell = self.file_list[idx].get("shell_pc")
        if shell is None:
            shell_path = os.path.join(sample["file_path"], "outer_crown.ply")
            mesh = o3d.io.read_triangle_mesh(shell_path)
            mesh.compute_vertex_normals()  # in case normals aren't stored in the file
            shell_points = np.asarray(mesh.vertices)
            shell_normals = np.asarray(mesh.vertex_normals)
            # fps_index = fpsample.bucket_fps_kdline_sampling(shell_points, 3000, h=5)
            # shell_points = shell_points[fps_index]
            # shell_normals = shell_normals[fps_index]
            shell = o3d.geometry.PointCloud()
            shell.points = o3d.utility.Vector3dVector(shell_points)
            shell.normals = o3d.utility.Vector3dVector(shell_normals)
            self.file_list[idx]["shell_pc"] = shell

        shellP = np.asarray(shell.points)
        shell_normals_original = np.asarray(shell.normals).copy()  # Add this line
        shell_min = np.min(shellP)
        shell_max = np.max(shellP)

        # normalize
        main_only, opposing_only, shell = (
            copy.deepcopy(main),
            copy.deepcopy(opposing),
            copy.deepcopy(shell),
        )
        main_only, opposing_only, shell, centroid, std_pc = (
            self.normalize_points_mean_std(main_only, opposing_only, shell)
        )
        shell_grid = self.file_list[idx].get("psr_grid")
        if shell_grid is None:
            shell_points_01 = (shellP - shell_min) / (shell_max + 1 - shell_min)
            shell_grid = self._compute_psr_for_sample(
                shell_points_01, shell_normals_original
            )
            self.file_list[idx]["psr_grid"] = shell_grid
        """""
        # sample from main
        patch_size_main = 5120
        positive_main_idx = np.arange(len(main_only))
        try:
            positive_selected_main_idx = np.random.choice(positive_main_idx, size=patch_size_main, replace=False)
        except ValueError:
            positive_selected_main_idx = np.random.choice(positive_main_idx, size=patch_size_main, replace=True)
        main_only_select = np.zeros([patch_size_main, main_only.shape[1]], dtype='float32')
        main_only_select[:] = main_only[positive_selected_main_idx, :]

        # sample from opposing
        patch_size_opposing = 5120
        positive_opposing_idx = np.arange(len(opposing_only))
        try:
            positive_selected_opposing_idx = np.random.choice(positive_opposing_idx, size=patch_size_opposing,
                                                              replace=False)
        except ValueError:
            positive_selected_opposing_idx = np.random.choice(positive_opposing_idx, size=patch_size_opposing,
                                                              replace=True)
        opposing_only_select = np.zeros([patch_size_opposing, opposing_only.shape[1]], dtype='float32')
        opposing_only_select[:] = opposing_only[positive_selected_opposing_idx, :]
     
        # sample from shell
        patch_size_shell = 1568
        positive_shell_idx = np.arange(len(shell))
        try:
            positive_selected_shell_idx = np.random.choice(positive_shell_idx, size=patch_size_shell, replace=False)
        except ValueError:
            positive_selected_shell_idx = np.random.choice(positive_shell_idx, size=patch_size_shell, replace=True)
        shell_select = np.zeros([patch_size_shell, shell.shape[1]], dtype='float32')
        shell_select[:] = shell[positive_selected_shell_idx, :]

        """ ""
        """""
        # sample from marginline
       
        patch_size_margin=300
        positive_margin_idx = np.arange(len(marginline_only))
        try:
           positive_selected_margin_idx = np.random.choice(positive_margin_idx, size=patch_size_margin, replace=False)
        except ValueError:    
           positive_selected_margin_idx = np.random.choice(positive_margin_idx, size=patch_size_margin, replace=True)
        marginline_only_select = np.zeros([patch_size_margin, marginline_only.shape[1]], dtype='float32')
        marginline_only_select[:] = marginline_only[positive_selected_margin_idx, :]
        
        """ ""

        # shell= open3d.geometry.sample_points_uniformly(shell, number_of_points=2048)
        # opposing_only= open3d.geometry.sample_points_uniformly(opposing_only, number_of_points=5120)
        # main_only= open3d.geometry.sample_points_uniformly(main_only, number_of_points=5120)

        # save through dataloader
        # X_train=np.multiply(shell_select,std_pc)+centroid
        # X_train_partial=np.multiply(main_only_select,std_pc)+centroid
        # groundtruth=np.concatenate((X_train,X_train_partial), axis=0)
        # np.save(os.path.join('Complet_2.npy'), X_train)
        # np.save(os.path.join('Partial_2.npy'), X_train_partial)
        """""
        shell_pc = torch.from_numpy(shell).float().unsqueeze(0)
        shell_sample = fps(shell_pc, 3072,device)
        data_gt =shell_sample.squeeze(0)
        antag_pc = torch.from_numpy(opposing_only).float().unsqueeze(0)
        antag_sample = fps(antag_pc, 5120,device)
        master_pc = torch.from_numpy(main_only).float().unsqueeze(0)
        master_sample = fps(master_pc, 5120,device)
        #data_partial = torch.concat((master_sample.squeeze(0), antag_sample.squeeze(0)))
        """ ""
        data_partial = torch.from_numpy(
            np.concatenate((main_only, opposing_only), axis=0)
        ).float()
        data_gt = torch.from_numpy(shell).float()
        min_gt = torch.from_numpy(np.asarray(shell_min)).float()
        max_gt = torch.from_numpy(np.asarray(shell_max)).float()
        value_centroid = torch.from_numpy(centroid).float()
        value_std_pc = torch.from_numpy(std_pc).float()
        shell_grid_gt = torch.from_numpy(np.asarray(shell_grid)).float()

        return (
            sample["taxonomy_id"],
            sample["model_id"],
            data_gt,
            data_partial,
            value_centroid,
            value_std_pc,
            shell_grid_gt,
            min_gt,
            max_gt,
        )

    def __len__(self):
        return len(self.file_list)
