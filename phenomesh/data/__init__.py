import os
import sys

import open3d as o3d
import cv2
import numpy as np

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
os.chdir(parent_dir)
sys.path.insert(0, parent_dir)

from phenomesh.io import MeshImporter

mesh_paths = {'2023-04-11_308': os.path.join('phenomesh', 'data', 'plot_mesh', '2023-04-11_308', '2023-04-11_308.ply')}
image_paths = {'DSC_0076.tif': os.path.join('phenomesh', 'data', 'plot_image', 'DSC_0076.tif')}


def get_image(image_path=mesh_paths['2023-04-11_308']):
    return cv2.imread(image_path)


def get_red_blue_gradient(w=256, h=256):
    gradient = np.zeros((h, w, 3), dtype=np.float32)

    red_gradient = np.linspace(0, 1, w)
    blue_gradient = np.linspace(0, 1, h)

    gradient[:, :, 0] = red_gradient
    gradient[:, :, 1] = 0.5  # Green channel
    gradient[:, :, 2] = blue_gradient[:, np.newaxis]  # Blue channel

    # Convert the floating point numbers to 8-bit integers [0-255]
    return ((gradient * 255).astype(np.uint8))


def get_mesh(mesh_path=mesh_paths['2023-04-11_308']):
    """
    Quick method for getting a mesh
    :param mesh_path: path to a triangle mesh
    :return: o3d.geometry.TriangleMesh()
    """
    return MeshImporter.import_mesh(mesh_path)

def get_subdivided_mesh(mesh_path=mesh_paths['2023-04-11_308'], iterations=2):
    mesh = MeshImporter.import_mesh(mesh_path)
    mesh = mesh.subdivide_loop(iterations)
    return mesh

def get_square():
    mesh = o3d.geometry.TriangleMesh()
    points = np.asarray([[1, 1, 0],
                        [-1, 1, 0],
                        [-1, -1, 0],
                        [1, -1, 0]])
    mesh.vertices = o3d.utility.Vector3dVector(points)
    mesh = _compute_normals(mesh)
    return mesh


def get_cube():
    cube = o3d.geometry.TriangleMesh.create_box(width=1, height=1, depth=1)
    cube = _compute_normals(cube)
    return cube


def get_cylinder():
    cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=0.045, height=1)
    cylinder = _compute_normals(cylinder)
    return cylinder


def _compute_normals(mesh):
    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()
    return mesh
