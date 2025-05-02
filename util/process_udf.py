"""
Code from SurfD https://github.com/Yzmblog/SurfD
"""
import torch
import open3d as o3d
import numpy as np
from typing import List, Tuple
from einops import repeat


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
    coords = torch.cat((coords, random_coords), dim=0)

    coords = torch.clip(coords, min=coords_range[0], max=coords_range[1])

    return coords

def compute_udf_and_gradients(
    mesh_o3d: o3d.geometry.TriangleMesh,
    queries: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    scene = o3d.t.geometry.RaycastingScene()
    vertices = np.asarray(mesh_o3d.vertices, dtype=np.float32)
    triangles = np.asarray(mesh_o3d.triangles, dtype=np.uint32)
    _ = scene.add_triangles(vertices, triangles)

    #signed_distance = scene.compute_signed_distance(query_point)
    closest_points = scene.compute_closest_points(queries.numpy())["points"]
    closest_points = torch.tensor(closest_points.numpy())

    q2p = queries - closest_points
    udf = torch.linalg.vector_norm(q2p, dim=-1)
    gradients = torch.nn.functional.normalize(q2p, dim=-1)

    return udf, gradients

def compute_udf_from_mesh(
    mesh_o3d: o3d.geometry.TriangleMesh,
    num_surface_points: int = 100_000,
    num_queries_on_surface: int = 10_000,
    queries_stds: List[float] = [0.003, 0.01, 0.1],
    num_queries_per_std: List[int] = [5_000, 4_000, 500, 500],
    coords_range: Tuple[float, float] = (-1.0, 1.0),
    input_queries = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pcd_o3d = mesh_o3d.sample_points_uniformly(number_of_points=num_surface_points)

    device = "cuda"
    pcd = get_tensor_pcd_from_o3d(pcd_o3d)[:, :3].to(device)
    if input_queries is not None:
        queries = input_queries
    else:
        queries = sample_points_around_pcd(
            pcd,
            queries_stds,
            num_queries_per_std,
            coords_range,
            device,
        )
    queries = queries.cpu()

    udf, gradients = compute_udf_and_gradients(mesh_o3d, queries)

    return queries, udf, gradients


def get_tensor_pcd_from_o3d(
    pcd_o3d: o3d.geometry.PointCloud,
    dtype: torch.dtype = torch.float,
) -> torch.Tensor:
    
    pcd_torch = torch.from_numpy(np.asarray(pcd_o3d.points)).to(dtype)

    if len(pcd_o3d.normals) > 0:
        normals_torch = torch.from_numpy(np.asarray(pcd_o3d.normals)).to(dtype)
        pcd_torch = torch.cat((pcd_torch, normals_torch), dim=-1)

    if len(pcd_o3d.colors) > 0:
        colors_torch = torch.from_numpy(np.asarray(pcd_o3d.colors)).to(dtype)
        pcd_torch = torch.cat((pcd_torch, colors_torch), dim=-1)

    return pcd_torch

def sample_udf_from_mesh(mesh_o3d, number_of_points, max_dist):
    
    pcd_o3d = mesh_o3d.sample_points_uniformly(number_of_points=number_of_points)

    surface = get_tensor_pcd_from_o3d(pcd_o3d)[:, :3]

    coords, udf, gradients = compute_udf_from_mesh(
        mesh_o3d,
        num_queries_on_surface=250_000,
        num_surface_points=100_000,
        # queries_stds=[0.003, 0.2, 0.05, 0.1],
        # num_queries_per_std=[25_000, 30_000, 20_000, 25_000, 500_000],
        # queries_stds=[0.1, 
        #               0.07, 
        #               0.05, 
        #               0.03, 
        #               0.01, 
        #               0.007, 
        #               0.005, 
        #               0.003, 
        #               0.001, 
        #               0.0005, 
        #               0.0001],
        # num_queries_per_std=[20_000, 
        #                 20_000, 
        #                 20_000, 
        #                 20_000, 
        #                 20_000, 
        #                 20_000, 
        #                 20_000, 
        #                 20_000, 
        #                 30_000, 
        #                 50_000, 
        #                 40_000,
        #                 250_000],
        queries_stds=[0.01,
                        0.005,
                        0.001,
                      ],
        num_queries_per_std=[75_000,
                             75_000,
                             75_000,
                        250_000],
    )

    coords = torch.cat((coords, surface), dim=0)
    udf = torch.cat((udf, torch.zeros(surface.shape[0], device=udf.device)), dim=0)
    gradients = torch.cat((gradients, torch.zeros_like(surface)), dim=0)
    perm_idxs = torch.randperm(coords.shape[0])
    coords = coords[perm_idxs]
    udf = udf[perm_idxs]
    gradients = gradients[perm_idxs]


    return surface.detach().cpu().numpy(), coords.detach().cpu().numpy(), udf.detach().cpu().numpy(), gradients.detach().cpu().numpy()