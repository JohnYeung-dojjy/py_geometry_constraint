"""
Simulate geometryConstraints function in Maya

referenced to:
    1. https://github.com/facebookresearch/pytorch3d/issues/193
    2. https://github.com/facebookresearch/pytorch3d/issues/1016
"""
from typing import List, Tuple

import numpy
import torch
from pytorch3d.structures.pointclouds import Pointclouds
from pytorch3d.structures.meshes import Meshes
from pytorch3d.loss.point_mesh_distance import _DEFAULT_MIN_TRIANGLE_AREA
from pytorch3d._C import point_face_dist_forward

def construct_pointcloud(points_to_be_projected, device='cpu') -> Pointclouds:
    if isinstance(points_to_be_projected, torch.Tensor):
        points_to_be_projected = points_to_be_projected.detach().to(device)
    elif isinstance(points_to_be_projected, numpy.ndarray):
        points_to_be_projected = torch.from_numpy(points_to_be_projected).to(device)
    else:
        points_to_be_projected = torch.tensor(points_to_be_projected, device=device)

    return Pointclouds([points_to_be_projected])

def construct_mesh(vtxs, faces, device='cpu')->Meshes:
    if isinstance(vtxs, torch.Tensor):
        vtxs = vtxs.detach().to(device)
    elif isinstance(vtxs, numpy.ndarray):
        vtxs = torch.from_numpy(vtxs).to(device)
    else:
        vtxs = torch.tensor(vtxs, device=device)

    if isinstance(faces, torch.Tensor):
        faces = faces.detach().to(device)
    elif isinstance(faces, numpy.ndarray):
        faces = torch.from_numpy(faces).to(device)
    else:
        faces = torch.tensor(faces, device=device)

    return Meshes([vtxs], [faces])

def geometry_constraint(point_cloud: Pointclouds, mesh_batch: Meshes) -> List[torch.Tensor]:
    """Simulate geometryConstraints function in Maya

    Args:
        vtxs (iterable): vertices of the mesh
        faces (iterable): faces of the mesh
        points_to_be_projected (iterable): points that will be projected onto the mesh constructed by `vtxs` and `faces`

    Returns:
        _type_: _description_
    """
    # Calculate the (`shortest_distance_to_mesh`, `shortest_distance_plane_id`) of our points' projection
    _, face_idx = point_to_face_distance(mesh_batch, point_cloud)

    # Calculate the projected points
    # Note: I don't have a nice way to vectorize the computation, but for my use case there are only < 100 points so simple for-loop is fine for me.
    points = point_cloud.points_packed()
    triangle_vertices_closest_to_points = mesh_batch.verts_packed()[mesh_batch.faces_packed()[face_idx]]

    projections = []
    for point, tri_vtxs in zip(points, triangle_vertices_closest_to_points):
        projections.append(point_projection_on_triangle(point, tri_vtxs))

    return projections


def point_to_face_distance(
    meshes: Meshes,
    pcls: Pointclouds,
    min_triangle_area: float = _DEFAULT_MIN_TRIANGLE_AREA,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    partially copied from pytorch3d.loss.point_mesh_face_distance

    Computes the distance between a pointcloud and a mesh within a batch.
    Given a pair `(mesh, pcl)` in the batch, we define the distance to be the
    sum of two distances, namely `point_face(mesh, pcl) + face_point(mesh, pcl)`

    `point_face(mesh, pcl)`: Computes the squared distance of each point p in pcl
        to the closest triangular face in mesh and averages across all points in pcl
    `face_point(mesh, pcl)`: Computes the squared distance of each triangular face in
        mesh to the closest point in pcl and averages across all faces in mesh.

    The above distance functions are applied for all `(mesh, pcl)` pairs in the batch
    and then averaged across the batch.

    Args:
        meshes: A Meshes data structure containing N meshes
        pcls: A Pointclouds data structure containing N pointclouds
        min_triangle_area: (float, defaulted) Triangles of area less than this
            will be treated as points/lines.

    Returns:
        dists: Minimum distance between
    """

    if len(meshes) != len(pcls):
        raise ValueError("meshes and pointclouds must be equal sized batches")

    # packed representation for pointclouds
    points = pcls.points_packed()  # (P, 3)
    points_first_idx = pcls.cloud_to_packed_first_idx()
    max_points = pcls.num_points_per_cloud().max().item()

    # packed representation for faces
    verts_packed = meshes.verts_packed()
    faces_packed = meshes.faces_packed()
    tris = verts_packed[faces_packed]  # (T, 3, 3)
    tris_first_idx = meshes.mesh_to_faces_packed_first_idx()

    # point to face distance: shape (P,)
    dists, idxs =  point_face_dist_forward(
        points, points_first_idx, tris, tris_first_idx, max_points, min_triangle_area
    )

    return dists, idxs

def point_projection_on_triangle(point: torch.Tensor, tri: torch.Tensor) -> torch.Tensor:
  """
  Compute and return the `projection_point` from `point` onto the triangular face formed by vertices `tri`.
  Args:
      point: FloatTensor of shape (3)
      tri: FloatTensor of shape (3, 3)
  Returns:
      p0: FloatTensor of shape (3)
  """
  a, b, c = tri.unbind(0)
  cross = torch.cross(b - a, c - a)
  # norm = cross.norm()
  normal = torch.nn.functional.normalize(cross, dim=0)


  # p0 is the projection of p onto the plane spanned by (a, b, c)
  # p0 = p + tt * normal, s.t. (p0 - a) is orthogonal to normal
  # => tt = dot(a - p, n)
  # tt = normal.dot(a) - normal.dot(point) # source code provided in the reference, not sure why it is un-matched with comment, result is same
  tt = (a-point).dot(normal)
  projection_point = point + tt * normal
  return projection_point