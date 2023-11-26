import os
import sys

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
