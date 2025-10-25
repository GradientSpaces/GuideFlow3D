import os
import torch
import trimesh
import numpy as np
import gc

def quad_to_triangle_mesh(F):
    """
    Converts a quad-dominant mesh into a pure triangle mesh by splitting quads into two triangles.

    Parameters:
        quad_mesh (trimesh.Trimesh): Input mesh with quad faces.

    Returns:
        trimesh.Trimesh: A new mesh with only triangle faces.
    """
    faces = F

    ### If already a triangle mesh -- skip
    if len(faces[0]) == 3:
        return F

    new_faces = []

    for face in faces:
        if len(face) == 4:  # Quad face
            # Split into two triangles
            new_faces.append([face[0], face[1], face[2]])  # Triangle 1
            new_faces.append([face[0], face[2], face[3]])  # Triangle 2
        else:
            print(f"Warning: Skipping non-triangle/non-quad face {face}")

    new_faces = np.array(new_faces)

    return new_faces

class Demo_Dataset(torch.utils.data.Dataset):
    def __init__(self, obj_path):
        super().__init__()

        self.obj_path = obj_path
        self.pc_num_pts = 100000

    
    def __len__(self):
        return 1

    def get_model(self):
        uid = os.path.basename(self.obj_path).split(".")[-2]
        mesh = trimesh.load(self.obj_path, force='mesh', process=False)
        vertices = mesh.vertices
        faces = mesh.faces

        bbmin = vertices.min(0)
        bbmax = vertices.max(0)
        center = (bbmin + bbmax) * 0.5
        scale = 2.0 * 0.9 / (bbmax - bbmin).max()
        vertices = (vertices - center) * scale
        mesh.vertices = vertices

        ### Make sure it is a triangle mesh -- just convert the quad
        mesh.faces = quad_to_triangle_mesh(faces)
        pc, _ = trimesh.sample.sample_surface(mesh, self.pc_num_pts) 

        result = {
                    'uid': uid
                }

        result['pc'] = torch.tensor(pc, dtype=torch.float32)
        result['vertices'] = mesh.vertices
        result['faces'] = mesh.faces

        return result

    def __getitem__(self, index):
        gc.collect()

        return self.get_model()