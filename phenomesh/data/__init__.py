import os
import sys

import open3d as o3d
import numpy as np

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
os.chdir(parent_dir)
sys.path.insert(0, parent_dir)

from phenomesh.io import import_mesh

mesh_paths = {'2023-04-11_308': os.path.join('phenomesh', 'data', 'plot_mesh', '2023-04-11_308', '2023-04-11_308.ply')}


def get_mesh(mesh_path=mesh_paths['2023-04-11_308']):
    """
    Quick method for getting a mesh
    :param mesh_path: path to a triangle mesh
    :return: o3d.geometry.TriangleMesh()
    """
    return import_mesh(mesh_path)

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
