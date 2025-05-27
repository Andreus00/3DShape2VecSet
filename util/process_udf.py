"""
Code from SurfD https://github.com/Yzmblog/SurfD
"""
import torch
import open3d as o3d
import numpy as np
from typing import List, Tuple
from einops import repeat
import trimesh


def sample_points_around_pcd(
    pcd: torch.Tensor,
    stds: List[float],
    num_points_per_std: List[int],
    coords_range: Tuple[float, float],
    device: str = "cpu",
) -> torch.Tensor:
    
    coords = torch.empty(0, 3).to(device)
    num_points_pcd = pcd.shape[0]

    for sigma, num_points in zip(stds, num_points_per_std[:-1]):
        mul = num_points // num_points_pcd

        if mul > 0:
            coords_for_sampling = repeat(pcd, "n d -> (n r) d", r=mul).to(device)
        else:
            coords_for_sampling = torch.empty(0, 3).to(device)

        still_needed = num_points % num_points_pcd
        if still_needed > 0:
            weights = torch.ones(num_points_pcd, dtype=torch.float).to(device)
            indices_random = torch.multinomial(weights, still_needed, replacement=False)
            pcd_random = pcd[indices_random].to(device)
            coords_for_sampling = torch.cat((coords_for_sampling, pcd_random), dim=0)

        offsets = torch.randn(num_points, 3).to(device) * sigma
        coords_i = coords_for_sampling + offsets

        coords = torch.cat((coords, coords_i), dim=0)

    random_coords = torch.rand(num_points_per_std[-1], 3).to(device)
    random_coords *= coords_range[1] - coords_range[0]
    random_coords += coords_range[0]

    coords = torch.clip(coords, min=coords_range[0], max=coords_range[1])
    random_coords = torch.clip(random_coords, min=coords_range[0], max=coords_range[1])

    return coords, random_coords

def compute_udf_and_gradients(
    vertices: np.ndarray,
    triangles: np.ndarray,
    queries: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    scene = o3d.t.geometry.RaycastingScene()
    _ = scene.add_triangles(vertices, triangles)

    #signed_distance = scene.compute_signed_distance(query_point)
    closest_points = scene.compute_closest_points(queries.numpy())["points"]
    closest_points = torch.tensor(closest_points.numpy())

    q2p = queries - closest_points
    udf = torch.linalg.vector_norm(q2p, dim=-1)
    gradients = torch.nn.functional.normalize(q2p, dim=-1)

    return udf, gradients

def compute_udf_from_mesh(
    mesh_trimesh: trimesh.Trimesh,
    num_surface_points: int = 100_000,
    num_queries_on_surface: int = 10_000,
    queries_stds: List[float] = [0.003, 0.01, 0.1],
    num_queries_per_std: List[int] = [5_000, 4_000, 500, 500],
    coords_range: Tuple[float, float] = (-1.0, 1.0),
    input_queries = None,
    device = "cuda",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    pcd, face_idx = mesh_trimesh.sample(count=num_surface_points, return_index=True)

    pcd = torch.asarray(pcd, device=device).float()
    if input_queries is not None:
        near_coords = input_queries
    else:
        near_coords, random_coords = sample_points_around_pcd(
            pcd,
            queries_stds,
            num_queries_per_std,
            coords_range,
            device,
        )
    near_coords = near_coords.cpu()
    random_coords = random_coords.cpu()

    vertices = np.asarray(mesh_trimesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh_trimesh.faces, dtype=np.uint32)
    udf_near, gradients_near = compute_udf_and_gradients(vertices, faces, near_coords)
    udf_rand, gradients_rand = compute_udf_and_gradients(vertices, faces, random_coords)

    return near_coords, udf_near, gradients_near, random_coords, udf_rand, gradients_rand


def sample_udf_from_mesh(mesh_trimesh: trimesh.Trimesh, number_of_points: int):
    
    pcd, face_idx = mesh_trimesh.sample(count=number_of_points, return_index=True) # .sample_points_uniformly(number_of_points=number_of_points)

    surface = torch.asarray(pcd).float()
    surface_grads = torch.asarray(mesh_trimesh.face_normals[face_idx]).float()


    coords_near, udf_near, gradients_near, coords_rand, udf_rand, gradients_rand = compute_udf_from_mesh(
        mesh_trimesh,
        num_queries_on_surface=250_000,
        num_surface_points=100_000,
        queries_stds=[0.05,
                        0.1,
                        0.001,
                      ],
        num_queries_per_std=[125_000,
                             125_000,
                             125_000,
                        250_000],
    )

    # udf_near = torch.cat((udf_near, torch.zeros(surface.shape[0], device=udf_near.device)), dim=0)
    # gradients_near = torch.cat((gradients_near, torch.zeros_like(surface)), dim=0)
    perm_idxs = torch.randperm(coords_near.shape[0])
    coords_near = coords_near[perm_idxs].detach().cpu().numpy()
    udf_near = udf_near[perm_idxs].detach().cpu().numpy()
    gradients_near = gradients_near[perm_idxs].detach().cpu().numpy()
    surface = surface.detach().cpu().numpy()


    return surface, surface_grads, coords_near, udf_near, gradients_near, coords_rand, udf_rand, gradients_rand