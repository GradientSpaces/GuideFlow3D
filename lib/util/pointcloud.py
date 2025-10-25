import numpy as np
import trimesh
import utils3d
import open3d as o3d

def convert_mesh_yup_to_zup(mesh):
    mesh.vertices = mesh.vertices @ np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
    return mesh

def voxelize_mesh(mesh_file, save_path):
    assert mesh_file.endswith('.ply') and save_path.endswith('.ply'), 'Voxelization only supports .ply files'
    
    mesh = o3d.io.read_triangle_mesh(mesh_file)

    # clamp vertices to the range [-0.5, 0.5]
    vertices = np.clip(np.asarray(mesh.vertices), -0.5 + 1e-6, 0.5 - 1e-6)
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(mesh, voxel_size=1/64, min_bound=(-0.5, -0.5, -0.5), max_bound=(0.5, 0.5, 0.5))
    vertices = np.array([voxel.grid_index for voxel in voxel_grid.get_voxels()])
    assert np.all(vertices >= 0) and np.all(vertices < 64), "Some vertices are out of bounds"
    vertices = (vertices + 0.5) / 64 - 0.5

    utils3d.io.write_ply(save_path, vertices)