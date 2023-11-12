import open3d as o3d
import numpy as np


class Mesh:
    def __init__(self, path=None, data=o3d.t.geometry.TriangleMesh()):
        self.path = path
        self.data = data

    def read(self, path):
        self.path = path
        self.data = o3d.t.io.read_triangle_mesh(filename=self.path)

    def write(self):
        o3d.t.io.write_triangle_mesh(filename=self.path, mesh=self.data)

    def view(self):
        o3d.visualization.draw_geometries([self.data.to_legacy()])


class PointCloud:
    def __init__(self, path=None, data=o3d.t.geometry.PointCloud()):
        self.path = path
        self.data = data

    def read(self, path):
        self.path = path
        self.data = o3d.t.io.read_point_cloud(filename=self.path)

    def write(self):
        o3d.t.io.write_point_cloud(filename=self.path, pointcloud=self.data)

    def view(self):
        o3d.visualization.draw_geometries([self.data.to_legacy()])





