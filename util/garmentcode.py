
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

def process_garment_worker(args, mean_body_mean, force_occupancy, max_dist):
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
    body_height = 171.0
    with open(body_info_path, 'r') as f:
        body_info = yaml.load(f, Loader=yaml.FullLoader)
        body_height = body_info.get('body', {}).get('height', 171.0)
    

    udf_path = os.path.join(subpath, f"{g}_udf.npz")
    if not os.path.exists(udf_path) or force_occupancy:
        # if os.path.exists(udf_path):
        #     try:
        #         with np.load(udf_path) as data:
        #             if "surface" in data and "points" in data and "labels" in data and "gradients" in data:
        #                 return {'model': model_file, 'point_path': udf_path, 'body_info_path': body_info_path,
        #                         'body_height': body_height, 'body_mean': mean_body_mean}
        #     except BadZipFile as e:
        #         print(f"Corrupted UDF file {udf_path}: {e}. Recomputing.")
        #         os.remove(udf_path)

        mesh_o3d: o3d.geometry.TriangleMesh = o3d.io.read_triangle_mesh(str(model_file))
        mesh_o3d.translate((-mean_body_mean[0].item(), -mean_body_mean[1].item(), -mean_body_mean[2].item()))
        mesh_o3d.scale(1 / body_height, center=np.zeros((3, 1)))

        surface, points, labels, gradients = sample_udf_from_mesh(mesh_o3d, max_dist)

        # import matplotlib.pyplot as plt
        # # Pick 10,000 random points
        # num_points_to_plot = min(10000, points.shape[0])
        # idxs = np.random.choice(points.shape[0], num_points_to_plot, replace=False)
        # sampled_points = points[idxs]
        # sampled_labels = labels[idxs]
        # sampled_labels[sampled_labels > 0.1] = 0.1
        # sampled_labels = 1 - sampled_labels / 0.1

        # # Plot in 3D using labels as color
        # fig = plt.figure(figsize=(10, 8))
        # ax = fig.add_subplot(111, projection='3d')
        # sc = ax.scatter(
        #     sampled_points[:, 0],
        #     sampled_points[:, 1],
        #     sampled_points[:, 2],
        #     c=sampled_labels,
        #     cmap='viridis',
        #     s=1
        # )
        # plt.colorbar(sc, label='Labels')
        # ax.set_xlabel('X')
        # ax.set_ylabel('Y')
        # ax.set_zlabel('Z')
        # plt.title('3D Point Cloud with Labels as Color')
        # plt.show()

        np.savez(udf_path, surface=surface, points=points, labels=labels, gradients=gradients)
        del surface, points, labels, gradients

    return {
        'model': model_file,
        'point_path': udf_path,
        'body_info_path': body_info_path,
        'body_height': body_height,
        'body_mean': mean_body_mean
    }


class GarmentCode(data.Dataset):

    def __init__(self, dataset_folder, split, force_occupancy=False, transform=None, sampling=True, num_samples=10_000, return_surface=True, surface_sampling=True, pc_size=4096, replica=100, max_dist=1.0):
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
                self.mesh_folders = self.mesh_folders[:split_idx][:1]
            elif self.split == "validation":
                self.mesh_folders = self.mesh_folders[split_idx:][:1]
                
        # Load mean body model
        self.mean_body_model: tri.Trimesh = tri.load(os.path.join(dataset_folder, 'neutral_body/mean_all.obj'))
        self.mean_body_mean = (self.mean_body_model.vertices * 100).mean(axis=0)

        # Parallen gpu running
        
        world_size = torch.cuda.device_count()
        print(f"Using {world_size} GPUs")

        with mp.get_context("spawn").Pool(processes=world_size) as pool:
            results = list(tqdm.tqdm(
                pool.imap_unordered(
                    partial(
                        process_garment_worker,
                        mean_body_mean=self.mean_body_mean,
                        force_occupancy=self.force_occupancy,
                        max_dist=self.max_dist,
                    ),
                    [(el, i % world_size) for i, el in enumerate(self.mesh_folders)]
                ),
                total=len(self.mesh_folders)
            ))
        # results = [self.process_garment(el) for el in tqdm.tqdm(self.mesh_folders, total=len(self.mesh_folders))]

        # Store processed results
        self.models = [res for res in results if res]

    def process_garment(self, subpath):
        """Processes a single garment, generating UDF if needed."""
        g = subpath.split('/')[-1]
        if not os.path.isdir(subpath):
            return None

        model_file = os.path.join(subpath, f"{g}_sim.ply")
        if not os.path.exists(model_file):
            print(f"Model {model_file} does not exist")
            return None

        body_info_path = os.path.join(subpath, f"{g}_body_measurements.yaml")
        body_height = 171.0  # Default height
        with open(body_info_path, 'r') as f:
            body_info = yaml.load(f, Loader=yaml.FullLoader)
            body_height = body_info.get('body', {}).get('height', 171.0)

        udf_path = os.path.join(subpath, f"{g}_udf.npz")
        if not os.path.exists(udf_path) or self.force_occupancy:
            if os.path.exists(udf_path):
                with np.load(udf_path) as data:
                    if "surface" in data and "points" in data and "labels" in data and "gradients" in data:
                        print(f"UDF already exists for {model_file}. Skipping.")
                        return {'model': model_file, 'point_path': udf_path, 'body_info_path': body_info_path,
                                'body_height': body_height, 'body_mean': self.mean_body_mean}

            mesh_o3d: o3d.geometry.TriangleMesh = o3d.io.read_triangle_mesh(str(model_file))
            mesh_o3d.translate((-self.mean_body_mean[0].item(), -self.mean_body_mean[1].item(), -self.mean_body_mean[2].item()))
            mesh_o3d.scale(1/body_height, center=np.zeros((3,1)))
            surface, points, labels, gradients = sample_udf_from_mesh(mesh_o3d, self.max_dist)
            # print(f"Saving UDF to {udf_path}")
            np.savez(udf_path, surface=surface, points=points, labels=labels, gradients=gradients)
            del surface, points, labels, gradients

        return {'model': model_file, 'point_path': udf_path, 'body_info_path': body_info_path,
                'body_height': body_height, 'body_mean': self.mean_body_mean}



    def __getitem__(self, idx):
        idx = idx % len(self.models)

        # model_path = self.models[idx]['model']
        point_path = self.models[idx]['point_path']

        try:
            with np.load(point_path) as data:
                points = data["points"]
                labels = data["labels"]
                surface = data["surface"]
                
        except Exception as e:
            print(e)
            print(point_path)

        if self.return_surface:
            surface = (surface - self.mean_body_mean) / self.models[idx]['body_height']
            if self.surface_sampling:
                idxs = np.random.default_rng().choice(surface.shape[0], self.pc_size, replace=False)
                surface = torch.from_numpy(surface[idxs]).float()
            else:
                surface = torch.from_numpy(surface.vertices).float()

        if self.sampling:
            idxs = np.random.default_rng().choice(points.shape[0], self.num_samples, replace=False)
            points = points[idxs]
            labels = labels[idxs]
        
        # Shuffle points and labels

        points = torch.from_numpy(points).float()
        labels = torch.from_numpy(labels).float()
        
        perm = torch.randperm(points.shape[0])
        points = points[perm]
        labels = labels[perm]

        if self.return_surface:
            return points, labels, surface, 0    # category is fixed for now
        else:
            return points, labels, 0 # category is fixed for now

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
