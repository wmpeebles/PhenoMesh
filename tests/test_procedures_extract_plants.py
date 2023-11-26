import os
import sys
import open3d as o3d
import numpy as np
import pytest
import logging

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
os.chdir(parent_dir)
sys.path.insert(0, parent_dir)

from phenomesh.data import get_mesh, get_square, get_cube, get_cylinder
from phenomesh.procedures.extract_plants import GeometryFilter as GF

logging.basicConfig(level=logging.DEBUG)

plot_mesh = get_mesh()

meshes = {'square': get_square(),
          'cube': get_cube(),
          'cylinder': get_cylinder()}


normal = np.asarray([0, 0, 1])


def test_calculate_angle_differences():
    normals = np.asarray([[0, 0, 1],
                          [0, 1, 0],
                          [1, 0, 0],
                          [0, 0, -1]])
    actual = GF._calculate_angle_differences(normal, normals)
    predicted = np.asarray([1, 0, 0, -1])
    assert actual.shape == predicted.shape
    for predicted_angle, actual_angle in zip(predicted.tolist(), actual.tolist()):
        assert predicted_angle == actual_angle, \
            f"predicted_angle {predicted_angle}, actual_angle {actual_angle}"


def test_calculate_angle_differences_2():
    normals = np.asarray([])
    actual = GF._calculate_angle_differences(normal, normals)
    predicted = 2
    assert actual == predicted


for mesh_name, mesh in meshes.items():
    gf = GF(mesh=mesh, max_angle=-0.75, within_radius=3)

    def test_mesh_to_pcd():
        pcd = gf._mesh_to_pcd(mesh)
        assert mesh.vertices == pcd.points
        assert mesh.vertex_normals == pcd.normals
        assert mesh.vertex_colors == pcd.colors

    def test_geometry_filter():
        new_mesh = gf.filter_mesh()
        assert id(mesh) != id(new_mesh)

meshes = {'plot_mesh': plot_mesh}


def test_vis_mesh():
    gf = GF(mesh=plot_mesh, max_angle=-0.9, within_radius=0.02)
    new_mesh = gf.filter_mesh()
    o3d.visualization.draw_geometries([plot_mesh])
    o3d.visualization.draw_geometries([new_mesh])


