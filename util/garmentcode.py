
import os
import glob
import random

import yaml 

import torch
from torch.utils import data

import numpy as np

from PIL import Image
import trimesh as tri
import mesh_to_sdf
import tqdm
import json

import open3d as o3d
from zipfile import BadZipFile
import multiprocessing as mp

from .process_udf import sample_udf_from_mesh, get_tensor_pcd_from_o3d

category_ids = {
    # todo: add category ids if necessary
}
import torch.multiprocessing as mp
from functools import partial



def process_garment_worker_meshbox_norm(args, mean_body_mean, force_occupancy, max_dist, body_model_normalization_alpha):
    """Processes a single garment on a specific GPU."""

    subpath, gpu_id = args
    torch.cuda.set_device(gpu_id)

    g = subpath.split('/')[-1]
    if not os.path.isdir(subpath):
        return None

    model_file = os.path.join(subpath, f"{g}_sim.ply")
    if not os.path.exists(model_file):
        print(f"Model {model_file} does not exist")
        return None

    body_info_path = os.path.join(subpath, f"{g}_body_measurements.yaml")
    
    udf_path = os.path.join(subpath, f"{g}_udf.npz")

    if not os.path.exists(udf_path) or force_occupancy:
        if os.path.exists(udf_path) and not force_occupancy:
            try:
                with np.load(udf_path) as data:
                    if "surface" in data and "points" in data and "udf" in data and "gradients" in data:
                        return {'model': model_file, 'point_path': udf_path, 'body_info_path': body_info_path}
            except BadZipFile as e:
                print(f"Corrupted UDF file {udf_path}: {e}. Recomputing.")
                os.remove(udf_path)


        mesh_o3d: o3d.geometry.TriangleMesh = o3d.io.read_triangle_mesh(str(model_file))
        shifts = (mesh_o3d.get_max_bound() + mesh_o3d.get_min_bound()) / 2
        mesh_o3d.translate((-shifts[0], -shifts[1], -shifts[2]))
        scale = (1 / np.abs(mesh_o3d.get_max_bound() - mesh_o3d.get_min_bound()).max()) * 1.9
        mesh_o3d.scale(scale, center=np.zeros((3, 1)))

        surface, points, udf, gradients = sample_udf_from_mesh(mesh_o3d, number_of_points=250_000, max_dist=max_dist)

        # import matplotlib.pyplot as plt
        # # Pick 10,000 random points
        # num_points_to_plot = min(100000, points.shape[0])
        # idxs = np.random.choice(points.shape[0], num_points_to_plot, replace=False)
        # sampled_udf = torch.asarray(udf[idxs])
        # sampled_points = torch.asarray(points[idxs])

        # from . import misc

        # sampled_labels = 1 - torch.clip(sampled_udf, 0, max_dist) / max_dist # misc.udf_to_labels(sampled_udf, max_dist)

        # for a, b in [(0.98, 1)]:
        #     m = torch.bitwise_and(sampled_labels >= a, sampled_labels <= b)
        #     interval_points = sampled_points[m]
        #     interval_labels = sampled_labels[m]
        #     # Plot in 3D using labels as color
        #     fig = plt.figure(figsize=(10, 8))
        #     ax = fig.add_subplot(111, projection='3d')
        #     sc = ax.scatter(
        #         interval_points[:, 0],
        #         interval_points[:, 1],
        #         interval_points[:, 2],
        #         c=interval_labels,
        #         cmap='viridis',
        #         s=1
        #     )
        #     plt.colorbar(sc, label='Labels')
        #     ax.set_xlabel('X')
        #     ax.set_ylabel('Y')
        #     ax.set_zlabel('Z')
        #     ax.set_xlim(-1, 1)
        #     ax.set_ylim(-1, 1)
        #     ax.set_zlim(-1, 1)
        #     plt.title('3D Point Cloud with Labels as Color')
        #     plt.show()

        np.savez(udf_path, surface=surface, points=points, udf=udf, gradients=gradients)
        del surface, points, udf, gradients

    return {
        'model': model_file,
        'point_path': udf_path,
        'body_info_path': body_info_path
    }



def process_garment_worker_body_model_norm(args, mean_body_mean, force_occupancy, max_dist, body_model_normalization_alpha):
    """Processes a single garment on a specific GPU."""
    subpath, gpu_id = args
    torch.cuda.set_device(gpu_id)

    g = subpath.split('/')[-1]
    if not os.path.isdir(subpath):
        return None

    model_file = os.path.join(subpath, f"{g}_sim.ply")
    if not os.path.exists(model_file):
        print(f"Model {model_file} does not exist")
        return None

    body_info_path = os.path.join(subpath, f"{g}_body_measurements.yaml")
    body_height = 171.99 # Default height
    with open(body_info_path, 'r') as f:
        body_info = yaml.load(f, Loader=yaml.FullLoader)
        body_height = body_info.get('body', {}).get('height', body_height)
    
    body_height = body_height * body_model_normalization_alpha
    
    udf_path = os.path.join(subpath, f"{g}_udf.npz")
    if not os.path.exists(udf_path) or force_occupancy:
        if os.path.exists(udf_path) and not force_occupancy:
            try:
                with np.load(udf_path) as data:
                    if "surface" in data and "points" in data and "udf" in data and "gradients" in data:
                        return {'model': model_file, 'point_path': udf_path, 'body_info_path': body_info_path,
                                'body_height': body_height, 'body_mean': mean_body_mean}
            except BadZipFile as e:
                print(f"Corrupted UDF file {udf_path}: {e}. Recomputing.")
                os.remove(udf_path)

        mesh_o3d: o3d.geometry.TriangleMesh = o3d.io.read_triangle_mesh(str(model_file))
        mesh_o3d.translate((-mean_body_mean[0].item(), -mean_body_mean[1].item(), -mean_body_mean[2].item()))
        mesh_o3d.scale(1 / body_height, center=np.zeros((3, 1)))

        surface, points, udf, gradients = sample_udf_from_mesh(mesh_o3d, number_of_points=100_000)

        np.savez(udf_path, surface=surface, points=points, udf=udf, gradients=gradients)
        del surface, points, udf, gradients

    return {
        'model': model_file,
        'point_path': udf_path,
        'body_info_path': body_info_path,
        'body_height': body_height,
        'body_mean': mean_body_mean
    }


class GarmentCode(data.Dataset):

    def __init__(self, dataset_folder, split, force_occupancy=False, transform=None, sampling=True, num_samples=10_000, return_surface=True, surface_sampling=True, pc_size=4096, replica=16, max_dist=0.1, body_model_normalization=False, body_model_normalization_alpha=0.5):
        self.pc_size = pc_size
        self.transform = transform
        self.num_samples = num_samples
        self.sampling = sampling
        self.split = split
        self.dataset_folder = dataset_folder
        self.return_surface = return_surface
        self.surface_sampling = surface_sampling
        self.replica = replica
        self.force_occupancy = force_occupancy
        self.max_dist = max_dist
        self.body_model_normalization = body_model_normalization
        self.body_model_normalization_alpha = body_model_normalization_alpha

        # Load split file
        train_val_test_path = os.path.join(dataset_folder, 'GarmentCodeData_v2_official_train_valid_test_data_split.json')
        if os.path.exists(train_val_test_path):
            with open(train_val_test_path, 'r') as f:
                train_test_val_split = json.load(f)
                if split not in train_test_val_split:
                    raise ValueError(f"Split {split} not found. Available: {train_test_val_split.keys()}")
                garments = [sample.replace("default_body", "default_body/data") for sample in train_test_val_split[split] if "default_body" in sample]
            # Build full paths
            self.mesh_folders = [os.path.join(dataset_folder, "GarmentCodeData_v2", garment) for garment in garments]
        else:
            garments_path = os.path.join(dataset_folder, "GarmentCodeData_v2", "garments_5000_0", "default_body", "data")
            self.mesh_folders = [os.path.join(garments_path, el) for el in os.listdir(garments_path)]
            split_idx = (len(self.mesh_folders) * 80) // 100
            if self.split == "training":
                self.mesh_folders = self.mesh_folders[:split_idx]
            elif self.split == "validation":
                self.mesh_folders = self.mesh_folders[split_idx:]
                
        # Load mean body model
        self.mean_body_model: tri.Trimesh = tri.load(os.path.join(dataset_folder, 'neutral_body/mean_all.obj'))
        self.mean_body_mean = (self.mean_body_model.vertices * 100).mean(axis=0)

        # Parallen gpu running
        
        world_size = torch.cuda.device_count()
        print(f"Using {world_size} GPUs")

        processing_func = process_garment_worker_body_model_norm if self.body_model_normalization else process_garment_worker_meshbox_norm
        
        with mp.get_context("spawn").Pool(processes=world_size) as pool:
            results = list(tqdm.tqdm(
                pool.imap_unordered(
                    partial(
                        processing_func,
                        mean_body_mean=self.mean_body_mean,
                        force_occupancy=self.force_occupancy,
                        max_dist=self.max_dist,
                        body_model_normalization_alpha=self.body_model_normalization_alpha
                    ),
                    [(el, i % world_size) for i, el in enumerate(self.mesh_folders)]
                ),
                total=len(self.mesh_folders)
            ))
        # Store processed results
        self.models = [res for res in results if res]


    def __getitem__(self, idx):
        idx = idx % len(self.models)

        # model_path = self.models[idx]['model']
        point_path = self.models[idx]['point_path']

        try:
            with np.load(point_path) as data:
                points = data["points"]
                udf = data["udf"]
                surface = data["surface"]
                
        except Exception as e:
            print(e)
            print(point_path)

        if self.return_surface:
            # surface = (surface - self.mean_body_mean) / self.models[idx]['body_height']
            if self.surface_sampling:
                idxs = np.random.default_rng().choice(surface.shape[0], self.pc_size, replace=False)
                surface = torch.from_numpy(surface[idxs]).float()
            else:
                surface = torch.from_numpy(surface.vertices).float()

        if self.sampling:
            idxs = np.random.default_rng().choice(points.shape[0], self.num_samples, replace=False)
            points = points[idxs]
            udf = udf[idxs]
        
        # Shuffle points and labels

        points = torch.from_numpy(points).float()
        udf = torch.from_numpy(udf).float()
        
        perm = torch.randperm(points.shape[0])
        points = points[perm]
        udf = udf[perm]

        if self.return_surface:
            return points, udf, surface, 0    # category is fixed for now
        else:
            return points, udf, 0 # category is fixed for now

    def __len__(self):
        if self.split != 'training':
            return len(self.models)
        else:
            return len(self.models) * self.replica
        

if __name__ == "__main__":
    dataset_path = "/home/andrea/Documents/PhD/Projects/SewingGaussians/GarmentCode/dataset_10_250303-18-02-40/"
    split = "train"  # or "test", "val" depending on your use case
    garment_dataset = GarmentCode(dataset_folder=dataset_path, split=split)

    print(f"Dataset loaded with {len(garment_dataset)} items.")
