import trimesh
import open3d as o3d
import torch
import numpy as np
from typing import List, Tuple

import matplotlib.pyplot as plt


def compute_udf_and_gradients(
    vertices: np.ndarray,
    triangles: np.ndarray,
    queries: torch.Tensor,
    device='cuda',
) -> Tuple[torch.Tensor, torch.Tensor]:
    scene = o3d.t.geometry.RaycastingScene()
    _ = scene.add_triangles(vertices, triangles)

    # compute the closest point on surface for queries
    closest_points = scene.compute_closest_points(queries.detach().cpu().numpy())["points"]
    closest_points = torch.tensor(closest_points.numpy(), device=device)

    q2p = queries - closest_points
    udf = torch.linalg.vector_norm(q2p, dim=-1)
    gradients = torch.nn.functional.normalize(q2p, dim=-1)

    return udf, gradients, closest_points


# Create a sphere using trimesh
sphere = trimesh.creation.icosphere(radius=1.0, subdivisions=3)
verts, faces = np.asarray(sphere.vertices, dtype=np.float32), np.asarray(sphere.faces, dtype=np.uint32)

# Sample 1000 random points close to the surface of the sphere
num_points = 1000
points_on_surface = sphere.sample(num_points)

# Add small random perturbations to move points slightly off the surface
perturbations = np.random.normal(scale=0.01, size=points_on_surface.shape)
points_near_surface = points_on_surface + perturbations

# Convert points_near_surface to a torch tensor
queries = torch.tensor(points_near_surface, dtype=torch.float32).to('cuda')

# Compute UDF and gradients
udf, gradients, closest_points = compute_udf_and_gradients(verts, faces, queries)

# Plot the sphere
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Extract vertices and faces
vertices = sphere.vertices
faces = sphere.faces

# Plot the mesh
# ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2], triangles=faces, color='lightblue', edgecolor='gray')
# # Add quiver plot for gradient directions
# ax.quiver(
#     queries[:, 0].detach().cpu().numpy(),
#     queries[:, 1].detach().cpu().numpy(),
#     queries[:, 2].detach().cpu().numpy(),
#     gradients[:, 0].detach().cpu().numpy(),
#     gradients[:, 1].detach().cpu().numpy(),
#     gradients[:, 2].detach().cpu().numpy(),
#     length=0.1,
#     color='red',
#     normalize=True
# )

# Plot the closest points on the surface
ax.scatter(
    closest_points[:, 0].detach().cpu().numpy(),
    closest_points[:, 1].detach().cpu().numpy(),
    closest_points[:, 2].detach().cpu().numpy(),
    color='green', s=10, label='Closest Points'
)

ax.scatter(
    verts[:, 0],
    verts[:, 1],
    verts[:, 2],
    color='blue', s=10, label='Closest Points'
)

# Set axis labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_ylim([-1.5, 1.5])
ax.set_xlim([-1.5, 1.5])
ax.set_zlim([-1.5, 1.5])

plt.show()