
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

from joblib import Parallel, delayed

category_ids = {
    # todo: add category ids if necessary
}

class GarmentCode(data.Dataset):
    # def __init__(self, dataset_folder, split, transform=None, sampling=True, num_samples=4096, return_surface=True, surface_sampling=True, pc_size=2048, replica=16):
        
    #     self.pc_size = pc_size

    #     self.transform = transform
    #     self.num_samples = num_samples
    #     self.sampling = sampling
    #     self.split = split

    #     self.dataset_folder = dataset_folder
    #     self.return_surface = return_surface
    #     self.surface_sampling = surface_sampling

    #     self.dataset_folder = dataset_folder

    #     # load split. garments are loaded in the format of "garments_5000_xx/default_body/data/rand_XXXXXXXXXXX"
    #     garments = None
    #     with open(os.path.join(dataset_folder, 'GarmentCodeData_v2_official_train_valid_test_data_split.json'), 'r') as f:
    #         train_test_val_split = json.load(f)
    #         if split not in train_test_val_split:
    #             raise ValueError(f"Split {split} not found in the train-test-val split file. Available splits are {train_test_val_split.keys()}")
    #         garments = [sample.replace("default_body", "default_body/data") for sample in train_test_val_split[split] if "default_body" in sample]
        
    #     # add the full path to the garment folders
    #     # the format will be "dataset_folder/GarmentCodeData_v2/garments_5000_xx/default_body/data/rand_XXXXXXXXXXX"
    #     self.mesh_folders = [os.path.join(dataset_folder, "GarmentCodeData_v2", garment) for garment in garments]

    #     self.mean_body_model: tri.Trimesh = tri.load(os.path.join(dataset_folder, 'neutral_body/mean_all.obj'))
    #     self.mean_body_mean = (self.mean_body_model.vertices * 100).mean(axis=0)

    #     pbar = tqdm.tqdm(self.mesh_folders, total=len(self.mesh_folders))

    #     self.models = []

    #     for subpath in pbar:
    #         # get the garment name
    #         g = subpath.split('/')[-1]
    #         assert os.path.isdir(subpath)
    #         model_file = os.path.join(subpath, f"{g}_sim.ply")
    #         if not os.path.exists(model_file):
    #             print(f"Model {model_file} does not exist")
    #             continue

    #         body_info_path = os.path.join(subpath, f"{g}_body_measurements.yaml")

    #         body_height = 171.0
    #         with open(body_info_path, 'r') as f:
    #             body_info = yaml.load(f, Loader=yaml.FullLoader)
    #             body_height = body_info.get('body', None).get('height', 171.0)

    #         udf_path = os.path.join(subpath, f"{g}_udf.npz")
    #         if not os.path.exists(udf_path):
    #             pbar.set_description(f"Generating UDF for {g}")
    #             # Load the model and generate UDF points
    #             model = tri.load(model_file)
    #             udf_inner_points, udf_inner_dists, udf_outer_points, udf_outer_dists = self.generate_udf_points(model=model, body_mean=self.mean_body_mean, body_height=body_height)  # todo: fix the default body
    #             np.savez(udf_path, vol_points=udf_inner_points, vol_label=udf_inner_dists, near_points=udf_outer_points, near_label=udf_outer_dists)
    #             del model, body_info, udf_inner_points, udf_inner_dists, udf_outer_points, udf_outer_dists
            
    #         self.models.append({'model': model_file, 'point_path': udf_path, 'body_info_path': body_info_path, 'body_height': body_height, 'body_mean': self.mean_body_mean})

    #     self.replica = replica

    def __init__(self, dataset_folder, split, transform=None, sampling=True, num_samples=4096, return_surface=True, surface_sampling=True, pc_size=2048, replica=16):
        self.pc_size = pc_size
        self.transform = transform
        self.num_samples = num_samples
        self.sampling = sampling
        self.split = split
        self.dataset_folder = dataset_folder
        self.return_surface = return_surface
        self.surface_sampling = surface_sampling
        self.replica = replica

        # Load split file
        with open(os.path.join(dataset_folder, 'GarmentCodeData_v2_official_train_valid_test_data_split.json'), 'r') as f:
            train_test_val_split = json.load(f)
            if split not in train_test_val_split:
                raise ValueError(f"Split {split} not found. Available: {train_test_val_split.keys()}")
            garments = [sample.replace("default_body", "default_body/data") for sample in train_test_val_split[split] if "default_body" in sample]

        # Build full paths
        self.mesh_folders = [os.path.join(dataset_folder, "GarmentCodeData_v2", garment) for garment in garments]

        # Load mean body model
        self.mean_body_model: tri.Trimesh = tri.load(os.path.join(dataset_folder, 'neutral_body/mean_all.obj'))
        self.mean_body_mean = (self.mean_body_model.vertices * 100).mean(axis=0)


        # Parallel processing using Joblib
        # num_workers = min(32, os.cpu_count())  # Use at most 32 workers
        # results = Parallel(n_jobs=num_workers, backend="loky")(
        #     delayed(self.process_garment)(subpath) for subpath in tqdm.tqdm(self.mesh_folders)
        # )

        # results = [self.process_garment(subpath) for subpath in tqdm.tqdm(self.mesh_folders)]
        
        # Parallel processing
        if False:
            with mp.Pool(8) as pool: # processes=mp.cpu_count()/3 * 2) as pool:
                results = list(tqdm.tqdm(pool.imap(self.process_garment, self.mesh_folders), total=len(self.mesh_folders)))
        else:
             results = [self.process_garment(el) for el in tqdm.tqdm(self.mesh_folders[:100], total=100)]

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
        if not os.path.exists(udf_path):
            print(f"Generating UDF for {model_file}")
            model = tri.load(model_file)
            udf_inner_points, udf_inner_dists, udf_outer_points, udf_outer_dists = self.generate_udf_points(
                model=model, body_mean=self.mean_body_mean, body_height=body_height
            )
            print(f"Saving UDF to {udf_path}")
            np.savez_compressed(udf_path, vol_points=udf_inner_points, vol_label=udf_inner_dists,
                     near_points=udf_outer_points, near_label=udf_outer_dists)
            print(f"UDF saved for {g}")
            del model, udf_inner_points, udf_inner_dists, udf_outer_points, udf_outer_dists

        return {'model': model_file, 'point_path': udf_path, 'body_info_path': body_info_path,
                'body_height': body_height, 'body_mean': self.mean_body_mean}


    def generate_udf_points(self, model: tri.Trimesh, body_mean, body_height, N=50000, threshold=0.01, sample_point_count=20000):
        '''
        Calculate UDF points for the given model
        '''
        # Normalize points based on body
        model.vertices -= body_mean
        model.vertices /= body_height

        # Generate UDF points
        points, sdf = mesh_to_sdf.sample_sdf_near_surface(model, number_of_points=N, surface_point_method='sample', sample_point_count=sample_point_count)
        
        udf = np.abs(sdf)

        inner_points = points[udf < threshold]
        inner_udf = udf[udf < threshold]

        outer_points = points[udf > threshold]
        outer_udf = udf[udf > threshold]

        return inner_points, inner_udf, outer_points, outer_udf


    def __getitem__(self, idx):
        idx = idx % len(self.models)

        model_path = self.models[idx]['model']
        point_path = self.models[idx]['point_path']

        try:
            with np.load(point_path) as data:
                vol_points = data['vol_points']
                vol_label = data['vol_label']
                near_points = data['near_points']
                near_label = data['near_label']
#                print(vol_points.shape, vol_label.shape, near_points.shape, near_label.shape)
                # exit()
        except Exception as e:
            print(e)
            print(point_path)

        # with open(point_path.replace('.npz', '.npy'), 'rb') as f:
        #     scale = np.load(f).item()

        if self.return_surface:
            surface = tri.load(model_path).vertices
#            print("surface vertices: ", surface.shape)
            surface = surface - self.mean_body_mean
            surface = surface / self.models[idx]['body_height']
            if self.surface_sampling:
                ind = np.random.default_rng().choice(surface.shape[0], self.pc_size, replace=False)
                surface = surface[ind]
            surface = torch.from_numpy(surface)

        if self.sampling:
            ind = np.random.default_rng().choice(vol_points.shape[0], self.num_samples, replace=False)
            vol_points = vol_points[ind]
            vol_label = vol_label[ind]

            ind = np.random.default_rng().choice(near_points.shape[0], self.num_samples, replace=False)
            near_points = near_points[ind]
            near_label = near_label[ind]
        else:
#            print("RESIZING")
            def pad_to_15000(arr: np.ndarray) -> np.ndarray:
                if arr.size >= 15000:
                    return arr[:15000]  # Truncate if necessary
                repeats = np.tile(arr, 15000 // arr.size + 1)  # Repeat elements
                return repeats[:15000]  # Trim to exactly 30,000 elements
            vol_points = pad_to_15000(vol_points)
            vol_label = pad_to_15000(vol_label)
            near_points = pad_to_15000(near_points)
            near_label = pad_to_15000(near_label)
        
        vol_points = torch.from_numpy(vol_points)
        vol_label = torch.from_numpy(vol_label).float()

        if self.split == 'training':
            near_points = torch.from_numpy(near_points)
            near_label = torch.from_numpy(near_label).float()

            points = torch.cat([vol_points, near_points], dim=0)
            labels = torch.cat([vol_label, near_label], dim=0)
        else:
            points = vol_points
            labels = vol_label

        if self.transform:
            surface, points = self.transform(surface, points)

#        print(points.shape, labels.shape, surface.shape)

        if self.return_surface:
            return points.float(), labels.float(), surface.float(), 0    # category is fixed for now
        else:
            return points.float(), labels.float(), 0 # category is fixed for now

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
