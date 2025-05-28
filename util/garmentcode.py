
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

from .process_udf import sample_udf_from_mesh

import torch.multiprocessing as mp
from functools import partial
import time
import matplotlib.pyplot as plt
import scipy.sparse

category_ids = {
    # todo: add category ids if necessary
}

# def build_pdf(A, normals, adj):
#     n_faces = len(normals)

#     # Step 1: Build sparse adjacency matrix (symmetric)
#     rows = np.concatenate([adj[:, 0], adj[:, 1]])
#     cols = np.concatenate([adj[:, 1], adj[:, 0]])
#     data = np.ones(len(rows))
#     adj_matrix = scipy.sparse.coo_matrix((data, (rows, cols)), shape=(n_faces, n_faces))

#     # Step 2: Compute dot product between each face and its neighbors
#     dot_products = adj_matrix.dot(normals)  # shape (n_faces, 3)
#     normal_mags = np.linalg.norm(dot_products, axis=1)
#     norm_normals = np.linalg.norm(normals, axis=1)
#     denom = norm_normals * normal_mags + 1e-8
#     cos_angles = np.einsum('ij,ij->i', normals, dot_products) / denom
#     cos_angles = np.clip(cos_angles, -1.0, 1.0)
#     angles = np.arccos(cos_angles)

#     # Step 3: Count neighbors per face
#     degree = np.asarray(adj_matrix.sum(axis=1)).flatten()
#     degree = np.maximum(degree, 1)

#     # Step 4: Average angle per face
#     mean_angle = angles / degree

#     # Step 5: Importance sampling weights
#     detail = np.maximum(mean_angle, 1e-6)
#     weights = A * detail
#     pdf = weights / weights.sum()

#     return pdf

def build_pdf(A, normals, adj, device):
    n_faces = len(normals)

    # Step 1: Build sparse adjacency matrix (symmetric)
    rows = torch.cat([adj[:, 0], adj[:, 1]])
    cols = torch.cat([adj[:, 1], adj[:, 0]])
    data = torch.ones(len(rows), device=device)
    adj_matrix = torch.sparse_coo_tensor(
        indices=torch.stack([rows, cols]),
        values=data,
        size=(n_faces, n_faces),
        device=device
    ).float()

    # Step 2: Compute dot product between each face and its neighbors
    dot_products = torch.sparse.mm(adj_matrix, normals)  # shape (n_faces, 3)
    normal_mags = torch.norm(dot_products, dim=1)
    norm_normals = torch.norm(normals, dim=1)
    denom = norm_normals * normal_mags + 1e-8
    cos_angles = (normals * dot_products).sum(dim=1) / denom
    cos_angles = cos_angles.clamp(-1.0, 1.0)
    angles = torch.arccos(cos_angles)

    # Step 3: Count neighbors per face
    degree = torch.sparse.sum(adj_matrix, dim=1).to_dense()
    degree = torch.clamp(degree, min=1.0)

    # Step 4: Average angle per face
    mean_angle = angles / degree

    # Step 5: Importance sampling weights
    detail = torch.clamp(mean_angle, min=1e-6)
    weights = A * detail
    pdf = weights / weights.sum()

    return pdf

# def importance_sampling(mesh, n_points=10_000):
#     A = mesh.area_faces
#     normals = mesh.face_normals
#     adj = mesh.face_adjacency

#     # Sampling function
#     def sample_points(pdf, n=1):
#         f_idx = np.random.choice(len(mesh.faces), size=n, p=pdf)
#         v = mesh.vertices[mesh.faces[f_idx]]
#         r = np.random.rand(n, 2)
#         sqrt_r1 = np.sqrt(r[:, 0])[:, None]
#         u = 1 - sqrt_r1
#         w = r[:, 1:2] * sqrt_r1
#         pts = u * v[:, 0] + w * v[:, 1] + (1 - u - w) * v[:, 2]

#         # Calculate triangle normal and use it as a gradient for the sampled points
#         grads = np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])
#         grads = grads / (np.linalg.norm(grads, axis=1, keepdims=True) + 1e-8)
#         return pts, grads

#     pdf = build_pdf(A, normals, adj)

#     # Sample
#     points, grads = sample_points(pdf=pdf, n=n_points)
    
#     return points, grads


def importance_sampling(mesh, n_points=10_000, device="cuda"):
    A = torch.tensor(mesh.area_faces, device=device).float()
    normals = torch.tensor(mesh.face_normals, device=device).float()
    adj = torch.tensor(mesh.face_adjacency, device=device).float()

    faces = torch.tensor(mesh.faces, device=device)
    vertices = torch.tensor(mesh.vertices, device=device)

    def sample_points(pdf, n=1):
        f_idx = torch.multinomial(pdf, num_samples=n, replacement=True)
        v = vertices[faces[f_idx]]  # shape: (n, 3, 3)

        r = torch.rand((n, 2), device=device)
        sqrt_r1 = torch.sqrt(r[:, :1])
        u = 1 - sqrt_r1
        w = r[:, 1:] * sqrt_r1
        pts = u * v[:, 0] + w * v[:, 1] + (1 - u - w) * v[:, 2]

        # Compute normals (gradients) via cross product
        grads = torch.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])
        grads = grads / (torch.norm(grads, dim=1, keepdim=True) + 1e-8)

        return pts, grads

    pdf = build_pdf(A, normals, adj, device=device)

    points, grads = sample_points(pdf, n=n_points)

    return points, grads


def process_garment_worker_meshbox_norm(args, mean_body_mean, force_occupancy, max_dist, body_model_normalization_alpha):
    """Processes a single garment on a specific GPU."""

    subpath, gpu_id = args
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if 'cuda' in device:
        torch.cuda.set_device(gpu_id)
        print('set device ', gpu_id)

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
                    if "surface" in data and "points_near" in data and "points_rand" in data and "udf" in data and "gradients" in data and "importance_points" in data:
                        return {'model': model_file, 'point_path': udf_path, 'body_info_path': body_info_path}
            except BadZipFile as e:
                print(f"Corrupted UDF file {udf_path}: {e}. Recomputing.")
                os.remove(udf_path)


        # mesh_o3d: o3d.geometry.TriangleMesh = o3d.io.read_triangle_mesh(str(model_file))
        # shifts = (mesh_o3d.get_max_bound() + mesh_o3d.get_min_bound()) / 2
        # mesh_o3d.translate((-shifts[0], -shifts[1], -shifts[2]))
        # scale = (1 / np.abs(mesh_o3d.get_max_bound() - mesh_o3d.get_min_bound()).max()) * 1.9
        # mesh_o3d.scale(scale, center=np.zeros((3, 1)))
        mesh_trimesh: tri.Trimesh = tri.load(str(model_file))
        b_min, b_max = mesh_trimesh.bounding_box.bounds[0], mesh_trimesh.bounding_box.bounds[1]
        shifts = (b_max + b_min) / 2
        mesh_trimesh = mesh_trimesh.apply_translation(-shifts)
        scale = (1 / np.abs(b_max - b_min).max()) * 1.9
        mesh_trimesh = mesh_trimesh.apply_scale(scale)

        # Check that scale is close to 1 and shifts are close to the origin
        b_min, b_max = mesh_trimesh.bounding_box.bounds[0], mesh_trimesh.bounding_box.bounds[1]
        shifts = (b_max + b_min) / 2
        scale = (1 / np.abs(b_max - b_min).max()) * 1.9
        if not (0.99 <= scale <= 1.01):
            print(f"Warning: Normalization Failed. Scale is not close to 1 (scale={scale}) for {model_file}")
        if not np.allclose(shifts, np.zeros_like(shifts), atol=1e-2):
            print(f"Warning: Normalization Failed. shifts are not close to origin (shifts={shifts}) for {model_file}")

        surface, surface_grads, points_near, udf_near, gradients_near, points_rand, udf_rand, gradients_rand = sample_udf_from_mesh(mesh_trimesh, number_of_points=250_000, device=device)

        # # Visualization: plot 10,000 points from each set (points_near, points_rand, surface)
        # fig = plt.figure(figsize=(18, 5))

        # # Plot points_near
        # ax1 = fig.add_subplot(131, projection='3d')
        # idxs_ = np.random.choice(points_near.shape[0], min(10_000, points_near.shape[0]), replace=False)
        # p_near = points_near[idxs_]
        # ax1.scatter(p_near[:, 0], p_near[:, 1], p_near[:, 2], s=1, c=udf_near[idxs_])
        # ax1.set_title('points_near')

        # # Plot points_rand
        # ax2 = fig.add_subplot(132, projection='3d')
        # idxs_ = np.random.choice(points_rand.shape[0], min(10_000, points_near.shape[0]), replace=False)
        # p_rand= points_rand[idxs_]
        # ax2.scatter(p_rand[:, 0], p_rand[:, 1], p_rand[:, 2], s=1, c=udf_rand[idxs_])
        # ax2.set_title('points_rand')

        # # Plot surface
        # ax3 = fig.add_subplot(133, projection='3d')
        # idxs_ = np.random.choice(surface.shape[0], min(10_000, points_near.shape[0]), replace=False)
        # p_sfc= surface[idxs_]
        # ax3.scatter(p_sfc[:, 0], p_sfc[:, 1], p_sfc[:, 2], s=1, c='red')
        # ax3.set_title('surface')

        # plt.tight_layout()
        # plt.show()
        # plt.pause(10)

        # mesh_trimesh = tri.load(str(model_file))
        # mesh_trimesh.vertices -= shifts
        # mesh_trimesh.vertices *= scale

        importance_points, importance_grad = importance_sampling(mesh_trimesh, n_points=50_000, device=device)

        np.savez(udf_path, surface=surface, surface_grads=surface_grads, importance_points=importance_points.detach().cpu(), importance_grad=importance_grad.detach().cpu(), points_near=points_near, \
                 points_rand=points_rand, udf_near=udf_near, udf_rand=udf_rand, gradients_near=gradients_near, \
                    gradients_rand=gradients_rand)
        del surface, points_near, udf_near, gradients_near, points_rand, udf_rand, gradients_rand

    return {
        'model': model_file,
        'point_path': udf_path,
        'body_info_path': body_info_path
    }

class GarmentCode(data.Dataset):

    def __init__(self, dataset_folder, split, force_occupancy=False, transform=None, sampling=True, num_samples=10_000, return_surface=True, surface_sampling=True, pc_size=4096, replica=1024, max_dist=0.1, body_model_normalization=False, body_model_normalization_alpha=0.5, random_samples_ratio=0.5, surface_samples_ratio=0.2):
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
        self.n_rnd_pts = int(random_samples_ratio * num_samples)
        self.n_sfc_pts = int(surface_samples_ratio * num_samples)
        self.n_near_pts = num_samples - (self.n_rnd_pts + self.n_sfc_pts)

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
                self.mesh_folders = self.mesh_folders[:split_idx]
                
        # Load mean body model
        self.mean_body_model: tri.Trimesh = tri.load(os.path.join(dataset_folder, 'neutral_body/mean_all.obj'))
        self.mean_body_mean = (self.mean_body_model.vertices * 100).mean(axis=0)

        # Parallen gpu running
        
        world_size = torch.cuda.device_count()
        if world_size > 0:
            print(f"Using {world_size} GPUs")

            processing_func = process_garment_worker_meshbox_norm
            
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
        else:
            world_size = os.cpu_count()
            print(f"Using {world_size} CPU")

            processing_func = process_garment_worker_meshbox_norm
            
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
                        [(el, -1) for i, el in enumerate(self.mesh_folders)]
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
                sfc = data["surface"]
                sfc_grads = data["surface_grads"]
                importance_points = data["importance_points"]
                points_near = data["points_near"]
                points_rand = data["points_rand"]
                udf_near = data["udf_near"]
                udf_rand = data["udf_rand"]
                importance_grad = data["importance_grad"]
                gradients_near = data["gradients_near"]
                gradients_rand = data["gradients_rand"]
                
        except Exception as e:
            print(e)
            print(point_path)

        if self.return_surface:
            if self.surface_sampling:
                idxs = np.random.default_rng().choice(sfc.shape[0], self.pc_size // 2, replace=False)
                idxs_importance = np.random.default_rng().choice(importance_points.shape[0], self.pc_size // 2, replace=False)
                
                surface = torch.cat([torch.from_numpy(sfc[idxs]), torch.from_numpy(importance_points[idxs_importance])]).float()
                grads = torch.cat([torch.from_numpy(sfc_grads[idxs]), torch.from_numpy(importance_grad[idxs_importance])]).float()
            else:
                surface = torch.cat([torch.from_numpy(sfc), torch.from_numpy(importance_points)]).float()
                grads = torch.cat([torch.from_numpy(sfc_grads), torch.from_numpy(importance_grad)]).float()

        if self.sampling:
            idxs_near = np.random.default_rng().choice(points_near.shape[0], self.n_near_pts, replace=False)
            idxs_rand = np.random.default_rng().choice(points_rand.shape[0], self.n_rnd_pts, replace=False)
            idxs_sfc = np.random.default_rng().choice(sfc.shape[0], self.n_sfc_pts, replace=False)
            points_near = points_near[idxs_near]
            udf_near = udf_near[idxs_near]
            points_rand = points_rand[idxs_rand]
            udf_rand = udf_rand[idxs_rand]
            points_sfc = sfc[idxs_sfc]
            udf_sfc = np.zeros((points_sfc.shape[0],))
            grads_near = gradients_near[idxs_near]
            grads_rand = gradients_rand[idxs_rand]
            grads_sfc = sfc_grads[idxs_sfc]
            points = np.concatenate([points_near, points_rand, points_sfc])
            udf = np.concatenate([udf_near, udf_rand, udf_sfc])
            grads = np.concatenate([grads_near, grads_rand, grads_sfc])
        else:
            points = np.concatenate([points_near, points_rand, sfc])
            udf = np.concatenate([udf_near, udf_rand, np.zeros((sfc.shape[0],))])
            grads = np.concatenate([gradients_near, gradients_rand, sfc_grads])
        
        # Shuffle points and labels

        points = torch.from_numpy(points).float()
        udf = torch.from_numpy(udf).float()
        grads = torch.from_numpy(grads).float()
        
        # perm = torch.randperm(points.shape[0])
        # points = points[perm]
        # udf = udf[perm]

        if self.return_surface:
            return points, udf, surface, grads, 0    # category is fixed for now
        else:
            return points, udf, grads, 0 # category is fixed for now

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
