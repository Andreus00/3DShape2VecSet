
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

import multiprocessing as mp


category_ids = {
    # todo: add category ids if necessary
}

class GarmentCode(data.Dataset):

    def __init__(self, dataset_folder, split, force_occupancy=False, transform=None, sampling=True, num_samples=8192, return_surface=True, surface_sampling=True, pc_size=4096, replica=1):
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

        # Parallel processing
        if self.force_occupancy:
            with mp.Pool(32) as pool: # processes=mp.cpu_count()/3 * 2) as pool:
                results = list(tqdm.tqdm(pool.imap(self.process_garment, self.mesh_folders), total=len(self.mesh_folders)))
        else:
             results = [self.process_garment(el) for el in tqdm.tqdm(self.mesh_folders, total=len(self.mesh_folders))]

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
            # print(f"Generating UDF for {model_file}")
            model = tri.load(model_file)

            random_points, near_points, udf_random, udf_near = self.generate_udf_points(
                model=model, body_mean=self.mean_body_mean, body_height=body_height
            )
            # print(f"Saving UDF to {udf_path}")
            np.savez(udf_path, random_points=random_points, near_points=near_points, udf_random=udf_random, udf_near=udf_near)
            del model, random_points, near_points, udf_random, udf_near
            # inner_points, outer_points = self.generate_udf_points(
            #     model=model, body_mean=self.mean_body_mean, body_height=body_height
            # )
            # # print(f"Saving UDF to {udf_path}")
            # np.savez(udf_path, inner_points=inner_points, outer_points=outer_points)
            # del model, inner_points, outer_points

        return {'model': model_file, 'point_path': udf_path, 'body_info_path': body_info_path,
                'body_height': body_height, 'body_mean': self.mean_body_mean}


    def generate_udf_points(self, model: tri.Trimesh, body_mean, body_height, N=100000, threshold=0.005, surface_sample_count=50000):
        '''
        Calculate UDF points for the given model
        '''
        
        # Normalize points based on body
        model.vertices -= body_mean
        model.vertices /= body_height

        # Generate UDF points
        query_points = mesh_to_sdf.surface_point_cloud.sample_uniform_points_in_unit_sphere(N)
        surface_points = tri.sample.sample_surface(model, surface_sample_count)[0]

        query_points = np.concatenate([query_points, surface_points,
                                       surface_points + np.random.normal(scale=0.1, size=(surface_sample_count, 3)),
                                       surface_points + np.random.normal(scale=0.01, size=(surface_sample_count, 3)),
                                       surface_points + np.random.normal(scale=0.005, size=(surface_sample_count, 3)),
                                       surface_points + np.random.normal(scale=0.00025, size=(surface_sample_count, 3)),], axis=0)
        sdf = mesh_to_sdf.mesh_to_sdf(mesh=model, query_points=query_points, sample_point_count=100000, surface_point_method="sample")

        udf = np.abs(sdf)

        return query_points[:N], query_points[N:], udf[:N], udf[N:]

    # def generate_udf_points(self, model: tri.Trimesh, body_mean, body_height, N=20000, threshold=0.005, surface_sample_count=50000):
    #     '''
    #     Calculate UDF points for the given model
    #     '''
        
    #     # Normalize points based on body
    #     model.vertices -= body_mean
    #     model.vertices /= body_height

    #     # Generate UDF points
    #     query_points = mesh_to_sdf.surface_point_cloud.sample_uniform_points_in_unit_sphere(N)
    #     surface_points = tri.sample.sample_surface(model, surface_sample_count)[0]

    #     query_points = np.concatenate([query_points, surface_points,
    #                                    surface_points + np.random.normal(scale=0.1, size=(surface_sample_count, 3)),
    #                                    surface_points + np.random.normal(scale=0.5, size=(surface_sample_count, 3)),
    #                                    surface_points + np.random.normal(scale=0.05, size=(surface_sample_count, 3)),
    #                                    surface_points + np.random.normal(scale=0.001, size=(surface_sample_count, 3)),
    #                                    surface_points + np.random.normal(scale=0.005, size=(surface_sample_count, 3)),
    #                                    surface_points + np.random.normal(scale=0.0005, size=(surface_sample_count, 3)),
    #                                    surface_points + np.random.normal(scale=0.00025, size=(surface_sample_count, 3)),], axis=0)
    #     sdf = mesh_to_sdf.mesh_to_sdf(mesh=model, query_points=query_points, sample_point_count=50000, surface_point_method="sample")

    #     udf = np.abs(sdf)

    #     mask = udf <= threshold

    #     inner_points = query_points[mask]
    #     outer_points = query_points[~mask]

    #     return inner_points, outer_points


    def __getitem__(self, idx):
        idx = idx % len(self.models)

        model_path = self.models[idx]['model']
        point_path = self.models[idx]['point_path']

        try:
            with np.load(point_path) as data:
                random_points = data["random_points"]
                udf_random = data["udf_random"]
                near_points = data["near_points"]
                udf_near = data["udf_near"]
                
        except Exception as e:
            print(e)
            print(point_path)

        if self.return_surface:
            surface = tri.load(model_path)
            surface.vertices = (surface.vertices - self.mean_body_mean) / self.models[idx]['body_height']
            if self.surface_sampling:
                surface = torch.from_numpy(tri.sample.sample_surface(surface, self.pc_size)[0]).float()
            else:
                surface = torch.from_numpy(surface.vertices).float()

        if self.sampling:
            ind_random_points = np.random.default_rng().choice(random_points.shape[0], self.num_samples//2, replace=False)
            ind_near_points = np.random.default_rng().choice(near_points.shape[0], self.num_samples//2, replace=False)
            points = np.concatenate([random_points[ind_random_points], near_points[ind_near_points]])

            labels = np.concatenate([udf_random[ind_random_points], udf_near[ind_near_points]])
        else:
            points = np.concatenate([random_points, near_points])
            labels = np.concatenate([udf_random, udf_near])
        
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
